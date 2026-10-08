from __future__ import annotations
import math, random, warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=FutureWarning)

ATTACKS=["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Reconnaissance","Shellcode","Worms"]
NORMAL="Normal"

@dataclass
class Config:
    mode:str="quick"; seeds:Tuple[int,...]=(11,); rounds:int=8; clients:int=4
    clients_scale:Tuple[int,...]=(4,10); dirichlet:Tuple[float,...]=(0.3,)
    local_epochs:int=1; batch:int=512; lr:float=1e-3; weight_decay:float=1e-4
    hidden:int=64; heads:int=4; layers:int=1; dropout:float=0.15
    malicious_fracs:Tuple[float,...]=(0.0,0.2); attacks:Tuple[str,...]=("label_flip","sign_flip","backdoor")
    aggregators:Tuple[str,...] = ("fedavg","fedprox","median","trimmed_mean","krum","trust_v2x")
    alphas:Tuple[float,...]=(0.3,); trim_ratio:float=0.2; prox_mu:float=0.01
    trust_threshold:float=0.25; reputation_decay:float=0.8; stale_eta:float=0.1
    max_train:int=45000; max_test:int=18000; zero_day_families:Tuple[str,...]=("Exploits","Fuzzers","Generic")
    zero_rounds:int=6; bootstrap:int=1000; workers:int=0

def config_for(mode):
    if mode=="full":
        return Config(mode="full", seeds=(11,29,47,71,101), rounds=20, clients=20,
          clients_scale=(4,10,20,50,100), dirichlet=(0.1,0.3,0.5,1.0), local_epochs=1,
          batch=1024, hidden=96, layers=2, malicious_fracs=(0,.1,.2,.3,.4),
          attacks=("label_flip","sign_flip","scaling","gaussian","backdoor"),
          max_train=175341, max_test=82332, zero_day_families=tuple(ATTACKS), zero_rounds=15)
    return Config()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def clean_labels(s:pd.Series):
    x=s.fillna(NORMAL).astype(str).str.strip().str.replace("-","",regex=False).str.lower()
    mapping={a.lower():a for a in ATTACKS}; mapping.update({"normal":NORMAL,"nan":NORMAL,"":NORMAL,"backdoors":"Backdoor"})
    return x.map(mapping).fillna(s.astype(str).str.strip().str.title())

def stratified_cap(df,n,seed):
    if n<=0 or len(df)<=n:return df.sample(frac=1,random_state=seed).reset_index(drop=True)
    parts=[]
    for _,g in df.groupby("attack_cat",dropna=False):
        k=max(1,round(n*len(g)/len(df))); parts.append(g.sample(min(k,len(g)),random_state=seed))
    out=pd.concat(parts).drop_duplicates()
    if len(out)>n: out=out.sample(n,random_state=seed)
    return out.sample(frac=1,random_state=seed).reset_index(drop=True)

def load_data(train_path,test_path,cfg,seed):
    tr=pd.read_csv(train_path,low_memory=False); te=pd.read_csv(test_path,low_memory=False)
    tr.columns=[c.strip().lower() for c in tr.columns]; te.columns=[c.strip().lower() for c in te.columns]
    if "attack_cat" not in tr: raise ValueError("CSV must include attack_cat")
    tr["attack_cat"]=clean_labels(tr["attack_cat"]); te["attack_cat"]=clean_labels(te["attack_cat"])
    tr=stratified_cap(tr,cfg.max_train,seed); te=stratified_cap(te,cfg.max_test,seed+1)
    drop=[c for c in ("id","label","attack_cat") if c in tr.columns]
    features=[c for c in tr.columns if c not in drop and c in te.columns]
    ytr=tr.attack_cat.values; yte=te.attack_cat.values
    Xtr=tr[features].replace([np.inf,-np.inf],np.nan); Xte=te[features].replace([np.inf,-np.inf],np.nan)
    cats=[c for c in features if Xtr[c].dtype=="object"]
    nums=[c for c in features if c not in cats]
    pre=ColumnTransformer([
      ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),nums),
      ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                       ("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False,min_frequency=2))]),cats)],
      sparse_threshold=0)
    Xtr=pre.fit_transform(Xtr).astype("float32"); Xte=pre.transform(Xte).astype("float32")
    classes=[NORMAL]+[a for a in ATTACKS if a in set(ytr)|set(yte)]
    enc={c:i for i,c in enumerate(classes)}
    return Xtr,np.array([enc[v] for v in ytr]),Xte,np.array([enc[v] for v in yte]),classes,pre

class TrustNet(nn.Module):
    """Compact latent-token TCN-Transformer for tabular network-flow records."""
    def __init__(self,d,c,h=64,drop=.15):
        super().__init__(); self.tokens=8; self.dim=max(16,h//2)
        self.inp=nn.Linear(d,self.tokens*self.dim)
        self.tcn=nn.Sequential(nn.Conv1d(self.dim,self.dim,3,padding=1,groups=self.dim),
            nn.Conv1d(self.dim,self.dim,1),nn.GELU(),nn.Dropout(drop),
            nn.Conv1d(self.dim,self.dim,3,padding=2,dilation=2,groups=self.dim),
            nn.Conv1d(self.dim,self.dim,1),nn.GELU())
        layer=nn.TransformerEncoderLayer(self.dim,4,self.dim*2,drop,batch_first=True,
            activation="gelu",norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,num_layers=1)
        self.gate=nn.Linear(self.dim,1);self.proj=nn.Sequential(nn.Linear(self.dim,h),nn.LayerNorm(h),nn.GELU())
        self.head=nn.Linear(h,c)
    def embed(self,x):
        z=self.inp(x).reshape(-1,self.tokens,self.dim);z=z+self.tcn(z.transpose(1,2)).transpose(1,2)
        z=self.transformer(z);w=torch.softmax(self.gate(z),dim=1);return F.normalize(self.proj((w*z).sum(1)),dim=1)
    def forward(self,x): return self.head(self.embed(x))

def loader(X,y,batch,shuffle=True):
    return DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y).long()),batch_size=batch,shuffle=shuffle)

def state_vec(state): return torch.cat([v.detach().float().cpu().reshape(-1) for v in state.values()])
def delta_state(local,global_): return OrderedDict((k,local[k].detach().cpu()-global_[k].detach().cpu()) for k in global_)
def add_delta(global_,delta,scale=1.): return OrderedDict((k,global_[k]+scale*delta[k]) for k in global_)

def partition_dirichlet(y,n,alpha,seed,min_size=20):
    rng=np.random.default_rng(seed); classes=np.unique(y)
    last=None
    for _ in range(25):
        out=[[] for _ in range(n)]
        for c in classes:
            ids=np.where(y==c)[0]; rng.shuffle(ids); p=rng.dirichlet(np.repeat(alpha,n)); cuts=(np.cumsum(p)*len(ids)).astype(int)[:-1]
            for j,a in enumerate(np.split(ids,cuts)):out[j].extend(a.tolist())
        if min(map(len,out))>=min_size:return [np.array(v,dtype=int) for v in out]
        last=out
    if len(y)<n*min_size:
        raise RuntimeError(f"Need at least {n*min_size} samples for {n} clients")
    out=last
    for target in sorted(range(n),key=lambda j:len(out[j])):
        while len(out[target])<min_size:
            donors=[j for j in range(n) if j!=target and len(out[j])>min_size]
            if not donors: raise RuntimeError("Could not repair client partition")
            donor=max(donors,key=lambda j:len(out[j]))
            take=min(min_size-len(out[target]),len(out[donor])-min_size)
            chosen=rng.choice(len(out[donor]),size=take,replace=False)
            chosen_set=set(np.atleast_1d(chosen).tolist())
            moved=[v for k,v in enumerate(out[donor]) if k in chosen_set]
            out[donor]=[v for k,v in enumerate(out[donor]) if k not in chosen_set]
            out[target].extend(moved)
    return [np.asarray(v,dtype=int) for v in out]

def local_train(global_state,X,y,classes,cfg,device,seed,prox=False,poison=None):
    seed_all(seed); yy=y.copy()
    if poison=="label_flip": yy=(yy+1)%classes
    model=TrustNet(X.shape[1],classes,cfg.hidden,cfg.dropout).to(device); model.load_state_dict(global_state)
    base={k:v.detach().clone().to(device) for k,v in global_state.items()}
    counts=np.bincount(yy,minlength=classes); w=len(yy)/(classes*np.maximum(counts,1)); lossfn=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    model.train()
    for _ in range(cfg.local_epochs):
        for xb,yb in loader(X,yy,cfg.batch):
            xb,yb=xb.to(device),yb.to(device)
            if poison=="backdoor":
                mask=torch.rand(len(xb),device=device)<.25; xb[mask,0]=6.; yb[mask]=0 if classes==2 else min(1,classes-1)
            opt.zero_grad(); loss=lossfn(model(xb),yb)
            if prox: loss += cfg.prox_mu/2*sum((p-base[n]).pow(2).sum() for n,p in model.named_parameters())
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
    d=delta_state(model.state_dict(),global_state)
    if poison=="sign_flip": d=OrderedDict((k,-5*v) for k,v in d.items())
    elif poison=="scaling": d=OrderedDict((k,10*v) for k,v in d.items())
    elif poison=="gaussian": d=OrderedDict((k,v+torch.randn_like(v)*max(v.std().item(),1e-3)*5) for k,v in d.items())
    return d

@torch.no_grad()
def predict(model,X,batch,device):
    model.eval(); logits=[]; embeds=[]
    for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)),batch_size=batch):
        xb=xb.to(device); logits.append(model(xb).cpu()); embeds.append(model.embed(xb).cpu())
    return torch.cat(logits).numpy(),torch.cat(embeds).numpy()

def multiclass_metrics(y,logits,classes):
    pred=logits.argmax(1); prob=torch.softmax(torch.tensor(logits),1).numpy(); ybin=label_binarize(y,classes=np.arange(classes))
    out={"accuracy":accuracy_score(y,pred),"macro_f1":f1_score(y,pred,average="macro",zero_division=0),
         "balanced_accuracy":balanced_accuracy_score(y,pred),"mcc":matthews_corrcoef(y,pred)}
    try: out["auprc"]=average_precision_score(y,prob[:,1]) if classes==2 else average_precision_score(ybin,prob,average="macro")
    except ValueError: out["auprc"]=np.nan
    try: out["auroc"]=roc_auc_score(y,prob[:,1]) if classes==2 else roc_auc_score(ybin,prob,average="macro",multi_class="ovr")
    except ValueError: out["auroc"]=np.nan
    cm=confusion_matrix(y,pred,labels=np.arange(classes)); fp=cm.sum(0)-np.diag(cm); tn=cm.sum()-cm.sum(0)-cm.sum(1)+np.diag(cm)
    out["fpr"]=float(fp[1]/max(fp[1]+tn[1],1)) if classes==2 else float(np.mean(fp/np.maximum(fp+tn,1)))
    return out
