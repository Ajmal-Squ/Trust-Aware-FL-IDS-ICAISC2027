from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

RESULTS = Path("results")
OUT = Path("figures"); OUT.mkdir(exist_ok=True)
raw = pd.read_csv(RESULTS / "threshold_raw.csv")
ev = pd.read_csv(RESULTS / "evidence_raw.csv")
attacks = ["label_flip", "sign_flip", "backdoor"]
labels = {"label_flip":"Label flip", "sign_flip":"Sign flip", "backdoor":"Backdoor"}

def save(name):
    plt.tight_layout(); plt.savefig(OUT / name, dpi=300, bbox_inches="tight"); plt.close()

for a in attacks:
    g=raw[raw.attack==a].groupby("threshold")["malicious_reject"].mean()*100
    plt.plot(g.index,g.values,marker="o",label=labels[a])
plt.xlabel("Trust threshold"); plt.ylabel("Malicious updates rejected (%)"); plt.ylim(0,105); plt.grid(axis="y",alpha=.25); plt.legend(frameon=False); save("Fig1_malicious_rejection.png")

for a in attacks:
    g=raw[raw.attack==a].groupby("threshold")["benign_reject"].mean()*100
    plt.plot(g.index,g.values,marker="o",label=labels[a])
plt.xlabel("Trust threshold"); plt.ylabel("Benign updates rejected (%)"); plt.ylim(0,100); plt.grid(axis="y",alpha=.25); plt.legend(frameon=False); save("Fig2_benign_rejection.png")

for a in attacks:
    g=raw[raw.attack==a].groupby("threshold")["screening_f1"].mean()
    plt.plot(g.index,g.values,marker="o",label=labels[a])
plt.xlabel("Trust threshold"); plt.ylabel("Malicious-update screening F1"); plt.ylim(0,1.05); plt.grid(axis="y",alpha=.25); plt.legend(frameon=False); save("Fig3_screening_f1.png")

metrics=[("Direction",lambda g:1-g.similarity),("Validation",lambda g:1-g.validation),("Norm",lambda g:g.outlier),("Combined",lambda g:1-g.trust)]
x=np.arange(len(metrics)); width=.22
fig,ax=plt.subplots(figsize=(6.8,4.2))
for i,a in enumerate(attacks):
    g=ev[ev.attack==a]; y=g.malicious.values
    vals=[roc_auc_score(y,fn(g)) for _,fn in metrics]
    ax.bar(x+(i-1)*width,vals,width=width,label=labels[a])
ax.set_xticks(x); ax.set_xticklabels([m[0] for m in metrics]); ax.set_ylabel("AUROC for malicious-update identification"); ax.set_ylim(0,1.05); ax.grid(axis="y",alpha=.25)
ax.legend(loc="lower center",bbox_to_anchor=(0.5,1.02),ncol=3,frameon=False)
fig.tight_layout(); fig.savefig(OUT/"Fig4_evidence_auroc.png",dpi=300,bbox_inches="tight"); plt.close(fig)

g=raw[raw.attack=="backdoor"].groupby("threshold")["attack_success_rate"].mean()*100
plt.plot(g.index,g.values,marker="o"); plt.xlabel("Trust threshold"); plt.ylabel("Backdoor ASR (%)"); plt.grid(axis="y",alpha=.25); save("Fig5_backdoor_asr.png")
print(f"Figures written to {OUT.resolve()}")
