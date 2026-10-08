from __future__ import annotations
import importlib.util, pathlib, sys, math, time, json
from collections import OrderedDict
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

BASE=str(pathlib.Path(__file__).with_name('metra_fl_binary_rerun.py'))
spec=importlib.util.spec_from_file_location('metra_base',BASE)
m=importlib.util.module_from_spec(spec); sys.modules['metra_base']=m; spec.loader.exec_module(m)
TR=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else pathlib.Path('data/UNSW_NB15_training-set.csv')
TE=pathlib.Path(sys.argv[2]) if len(sys.argv)>2 else pathlib.Path('data/UNSW_NB15_testing-set.csv')
OUT=pathlib.Path(sys.argv[3]) if len(sys.argv)>3 else pathlib.Path('results'); OUT.mkdir(parents=True, exist_ok=True)
SEEDS=(11,29,47); ATTACKS=('label_flip','sign_flip','backdoor'); THRESHOLDS=(.35,.45,.55,.65,.75)
DEVICE=torch.device('cpu')

def aggregate_threshold(deltas,sizes,trust,outlier,threshold):
    vec=torch.stack([m.state_vec(d) for d in deltas])
    accepted=trust>=threshold
    if not accepted.any(): accepted[np.argmax(trust)]=True
    weights=trust*np.asarray(sizes)*accepted
    weights=weights/(weights.sum()+1e-12)
    med=vec.median(0).values; mad=(vec-med).abs().median(0).values+1e-9
    safe=torch.maximum(torch.minimum(vec,med+3*mad),med-3*mad)
    agg=(safe*torch.tensor(weights,dtype=torch.float32).unsqueeze(1)).sum(0)
    template=deltas[0]; out=OrderedDict(); pos=0
    for k,v in template.items():
        n=v.numel(); out[k]=agg[pos:pos+n].reshape(v.shape).to(v.dtype); pos+=n
    return out,accepted

def run_case(seed,attack):
    cfg=m.config_for('full'); cfg.max_train=175341; cfg.max_test=82332; cfg.clients=20; cfg.batch=1024
    cfg.hidden=96; cfg.dropout=.15; cfg.local_epochs=1; cfg.trust_threshold=.55
    Xtr,ymtr,Xte,ymte,names,pre=m.load_data(TR,TE,cfg,seed)
    ytr=(ymtr!=0).astype(np.int64); yte=(ymte!=0).astype(np.int64)
    Xdev,Xval,ydev,yval=train_test_split(Xtr,ytr,test_size=.2,random_state=seed,stratify=ytr)
    m.seed_all(seed); parts=m.partition_dirichlet(ydev,20,.3,seed)
    model=m.TrustNet(Xdev.shape[1],2,cfg.hidden,cfg.dropout).to(DEVICE)
    global_state=OrderedDict((k,v.detach().cpu()) for k,v in model.state_dict().items())
    reps=np.repeat(.5,20); rng=np.random.default_rng(seed); malicious=set(rng.choice(20,4,replace=False).tolist())
    ridx=rng.choice(len(Xval),min(1500,len(Xval)),replace=False); Xref,yref=Xval[ridx],yval[ridx]
    deltas=[]; sizes=[]
    for i,idx in enumerate(parts):
        d=m.local_train(global_state,Xdev[idx],ydev[idx],2,cfg,DEVICE,seed+i,False,attack if i in malicious else None)
        deltas.append(d); sizes.append(len(idx))
    vec=torch.stack([m.state_vec(d) for d in deltas]); center=vec.median(0).values
    norms=vec.norm(dim=1); med=norms.median(); mad=(norms-med).abs().median()+1e-9
    sim=F.cosine_similarity(vec,center.unsqueeze(0)).add(1).div(2).numpy()
    out=np.clip(((norms-med).abs()/(3*mad)).numpy(),0,1)
    base_model=m.TrustNet(Xdev.shape[1],2,cfg.hidden,cfg.dropout).to(DEVICE); base_model.load_state_dict(global_state)
    base_loss=F.cross_entropy(torch.tensor(m.predict(base_model,Xref,cfg.batch,DEVICE)[0]),torch.tensor(yref)).item()
    vals=[]
    for d in deltas:
        tmp=m.TrustNet(Xdev.shape[1],2,cfg.hidden,cfg.dropout).to(DEVICE); tmp.load_state_dict(m.add_delta(global_state,d))
        loss=F.cross_entropy(torch.tensor(m.predict(tmp,Xref,cfg.batch,DEVICE)[0]),torch.tensor(yref)).item()
        vals.append(float(1/(1+math.exp(np.clip((loss-base_loss)*3,-20,20)))))
    val=np.asarray(vals)
    trust=np.clip(.30*sim+.35*val+.15*reps+.20*(1-out),0,1)
    labels=np.array([1 if i in malicious else 0 for i in range(20)]) # 1 malicious
    rows=[]
    for tau in THRESHOLDS:
        agg,accepted=aggregate_threshold(deltas,sizes,trust,out,tau)
        rejected=~accepted
        tp=int(np.sum(rejected & (labels==1))); fp=int(np.sum(rejected & (labels==0)))
        fn=int(np.sum(accepted & (labels==1))); tn=int(np.sum(accepted & (labels==0)))
        prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); sf1=2*prec*rec/max(prec+rec,1e-12)
        malrej=tp/max(tp+fn,1); benrej=fp/max(fp+tn,1)
        eval_model=m.TrustNet(Xdev.shape[1],2,cfg.hidden,cfg.dropout).to(DEVICE)
        eval_model.load_state_dict(m.add_delta(global_state,agg))
        logits,_=m.predict(eval_model,Xte,cfg.batch,DEVICE); met=m.multiclass_metrics(yte,logits,2)
        asr=np.nan
        if attack=='backdoor':
            trigger=Xte.copy(); trigger[:,0]=6.; pp=m.predict(eval_model,trigger,cfg.batch,DEVICE)[0].argmax(1)
            asr=float((pp[yte==1]==0).mean())
        rows.append(dict(seed=seed,attack=attack,threshold=tau,malicious_reject=malrej,benign_reject=benrej,
                         screening_precision=prec,screening_recall=rec,screening_f1=sf1,attack_success_rate=asr,
                         accuracy=met['accuracy'],macro_f1=met['macro_f1'],balanced_accuracy=met['balanced_accuracy'],mcc=met['mcc'],fpr=met['fpr'],
                         mean_trust_malicious=float(trust[labels==1].mean()),mean_trust_benign=float(trust[labels==0].mean()),
                         min_trust_malicious=float(trust[labels==1].min()),max_trust_malicious=float(trust[labels==1].max()),
                         min_trust_benign=float(trust[labels==0].min()),max_trust_benign=float(trust[labels==0].max())))
    evidence=pd.DataFrame(dict(seed=seed,attack=attack,client=np.arange(20),malicious=labels,trust=trust,similarity=sim,validation=val,outlier=out))
    return rows,evidence

allrows=[]; allevidence=[]
for seed in SEEDS:
    for attack in ATTACKS:
        t=time.time(); print('RUN',seed,attack,flush=True)
        rows,ev=run_case(seed,attack); allrows.extend(rows); allevidence.append(ev)
        pd.DataFrame(allrows).to_csv(OUT/'threshold_raw.csv',index=False)
        pd.concat(allevidence,ignore_index=True).to_csv(OUT/'evidence_raw.csv',index=False)
        print('DONE',seed,attack,'sec',round(time.time()-t,1),flush=True)
raw=pd.DataFrame(allrows); ev=pd.concat(allevidence,ignore_index=True)
summary=raw.groupby(['attack','threshold']).agg({c:['mean','std'] for c in ['malicious_reject','benign_reject','screening_precision','screening_recall','screening_f1','attack_success_rate','accuracy','macro_f1','balanced_accuracy','mcc','fpr']}).reset_index()
summary.columns=['_'.join(x).strip('_') for x in summary.columns]; summary.to_csv(OUT/'threshold_summary.csv',index=False)
ev.groupby(['attack','malicious'])[['trust','similarity','validation','outlier']].agg(['mean','std','min','max']).to_csv(OUT/'evidence_summary.csv')
print('COMPLETE',OUT,flush=True)
