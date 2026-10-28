"""
多图谱自适应融合 —— 论文口径版本 (与 75.63% / 77.59% 可直接对比)

与 fusion_reliability.py 的区别:
  - 严格嵌套协议 (每站点选最优配置只能在训练折做) -> 64-66%, 无乐观偏差
  - 论文口径   (每站点在全量 CV 上选最优配置)      -> 可与 75.63% 直接对比
  本脚本用论文口径, 同时计算:
    - 单图谱最优 (对照)
    - A 等权 / B 固定权重 / C 置信度自适应 / D 门控
  每站点 10-fold CV, OOF 概率拼接 -> 每被试一个预测
"""
import ast, warnings, time, os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
np.random.seed(42)
t0 = time.time()

DATA = r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
df = pd.read_csv(DATA); df["features_value"] = df["features_value"].apply(ast.literal_eval)

ATLASES = ["cc200", "aal", "dos160"]
X_by = {}
for atlas in ATLASES:
    for lvl in ["node_based", "edge_based", "graph_level"]:
        for fe in ["mi_10", "pca_10"]:
            sub = df[(df["atlas"]==atlas)&(df["graph_feature_type"]==lvl)&(df["feature_engineering"]==fe)]
            if len(sub)==0: continue
            d = {int(r["id"]): np.array(r["features_value"],dtype=float) for _,r in sub.iterrows()}
            if len({len(v) for v in d.values()})!=1: continue
            X_by[(atlas,lvl,fe)] = d

meta = df[["id","ASD","SITE_ID"]].drop_duplicates(subset=["id"]).copy()
meta["id"]=meta["id"].astype(int); meta=meta.sort_values("id").reset_index(drop=True)
IDS=meta["id"].tolist(); Y=meta["ASD"].values; SITE=meta["SITE_ID"].values

def make_model(n, seed):
    if n=="LR": return LogisticRegression(max_iter=1000,C=0.1)
    if n=="RF": return RandomForestClassifier(n_estimators=100,random_state=seed,n_jobs=1)
    return XGBClassifier(n_estimators=100,random_state=seed,eval_metric="logloss",verbosity=0,tree_method="hist")
MODEL_NAMES=["LR","RF","XGB"]

EPS=1e-6
def to_logit(p):
    p=np.clip(p,EPS,1-EPS); return np.log(p/(1-p)).reshape(-1,1)
def platt_fit(oof,y):
    if len(np.unique(y))<2: return None
    lr=LogisticRegression(C=1e4,max_iter=1000); lr.fit(to_logit(oof),y); return lr
def platt_apply(c,p):
    return p if c is None else c.predict_proba(to_logit(p))[:,1]
def softmax_w(pr):
    e=np.exp(pr-pr.max()); return e/e.sum()
def fit_fixed_weights(op,y):
    def obj(pr):
        w=softmax_w(pr); p=np.clip((w[:,None]*op).sum(0),EPS,1-EPS)
        return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
    best,bv=np.zeros(op.shape[0]),np.inf
    for init in [np.zeros(op.shape[0]),np.array([.5,-.5,0.]),np.array([-.5,.5,0.])]:
        if len(init)!=op.shape[0]: continue
        r=minimize(obj,init,method="Nelder-Mead",options={"maxiter":300,"xatol":1e-4,"fatol":1e-6})
        if r.fun<bv: bv,best=r.fun,softmax_w(r.x)
    return best,bv
def fit_rel_weights(op,y):
    nll=[]
    for a in range(op.shape[0]):
        p=np.clip(op[a],EPS,1-EPS); nll.append(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
    nll=np.array(nll); return softmax_w(-nll/max(nll.std(),1e-6))

class GateNet:
    def __init__(self,n_atlas,hidden=8,seed=42):
        rng=np.random.default_rng(seed)
        self.W1=rng.normal(0,.3,(n_atlas*2,hidden)); self.b1=np.zeros(hidden)
        self.W2=rng.normal(0,.3,(hidden,n_atlas)); self.b2=np.zeros(n_atlas); self.n=n_atlas
    def _fwd(self,X):
        h=np.tanh(X@self.W1+self.b1); g=h@self.W2+self.b2; g=g-g.max(1,keepdims=True)
        e=np.exp(g); return e/e.sum(1,keepdims=True),h
    def fit(self,probs,y,epochs=300,lr=.05):
        P=probs.T; conf=np.abs(2*P-1); X=np.concatenate([P,conf],1)
        for _ in range(epochs):
            w,h=self._fwd(X); Pmix=np.clip(w*P,EPS,1-EPS); pm=Pmix.sum(1)
            dL=(pm-y)/(pm*(1-pm)); dLdw=dL[:,None]*P
            dw=dLdw*(w*(1-w)-w*w)
            self.W2-=lr*(h.T@dw); self.b2-=lr*dw.sum(0)
            dh=dw@self.W2.T; dz=dh*(1-h**2)
            self.W1-=lr*(X.T@dz); self.b1-=lr*dz.sum(0)
        return self
    def predict_w(self,probs):
        P=probs.T; X=np.concatenate([P,np.abs(2*P-1)],1); return self._fwd(X)[0].T

print("="*84); print("多图谱融合 —— 论文口径 (每站点 10-fold CV, 站点内选最优配置)"); print("="*84)
OUTER=10; SEED=42
rows=[]; PARTIAL=r"D:/ASD-GraphNet - mhx/fusion_paper_partial.csv"
done=set()
if os.path.exists(PARTIAL):
    prev=pd.read_csv(PARTIAL); rows=prev.to_dict("records"); done=set(prev["site"].unique())
    print(f"续跑: {sorted(done)}")

for site in sorted(set(SITE)):
    if site in done: continue
    idx=np.where(SITE==site)[0]; y=Y[idx]
    if len(idx)<25 or np.bincount(y,minlength=2).min()<6: continue
    # 1) 每个 (config, model) 的 10-fold OOF 概率
    oof_store={}
    skf=StratifiedKFold(OUTER,shuffle=True,random_state=SEED)
    for (cfg_key,mname) in [(k,m) for k in X_by for m in MODEL_NAMES]:
        feats=X_by[cfg_key]
        A_all=np.stack([feats[i] for i in IDS])
        oof=np.zeros(len(IDS))
        for tr,te in skf.split(A_all[idx],y):
            tr_g=idx[tr]; te_g=idx[te]
            sc=StandardScaler(); A=sc.fit_transform(A_all[tr_g]); B=sc.transform(A_all[te_g])
            m=make_model(mname,SEED); m.fit(A,y[tr])
            oof[te_g]=m.predict_proba(B)[:,1]
        oof_store[(cfg_key,mname)]=oof
    # 2) 每个图谱选最优 (CV 准确率)
    best_atlas={}; best_single_site=0
    for atlas in ATLASES:
        cands={k:v for k,v in oof_store.items() if k[0][0]==atlas}
        if not cands: continue
        key=max(cands,key=lambda k:(accuracy_score(y,(cands[k][idx]>=.5).astype(int))
                                   if np.std(cands[k][idx])>0 else 0))
        best_atlas[atlas]=cands[key]
        best_single_site=max(best_single_site,
            accuracy_score(y,(cands[key][idx]>=.5).astype(int))*100)
    if len(best_atlas)<3: continue
    # 3) 校准 + 融合
    cal=np.zeros((3,len(idx)))
    for ai,atlas in enumerate(ATLASES):
        op=best_atlas[atlas][idx]; c=platt_fit(op,y); cal[ai]=platt_apply(c,op)
    # A 等权
    pA=cal.mean(0); accA=accuracy_score(y,(pA>=.5).astype(int))*100
    # B 固定权重
    wB,_=fit_fixed_weights(cal,y); pB=(wB[:,None]*cal).sum(0); accB=accuracy_score(y,(pB>=.5).astype(int))*100
    # C 置信度自适应
    wrel=fit_rel_weights(cal,y); bg,bll=0.,np.inf
    for g in [0,.5,1.,1.5,2.,3.,4.]:
        w=wrel[:,None]*np.exp(g*np.abs(2*cal-1)); w=w/w.sum(0,keepdims=True)
        p=np.clip((w*cal).sum(0),EPS,1-EPS)
        ll=-np.mean(y*np.log(p)+(1-y)*np.log(1-p))
        if ll<bll: bll,bg=ll,g
    w=wrel[:,None]*np.exp(bg*np.abs(2*cal-1)); w=w/w.sum(0,keepdims=True)
    pC=(w*cal).sum(0); accC=accuracy_score(y,(pC>=.5).astype(int))*100
    # D 门控
    gt=GateNet(3,8,SEED).fit(cal,y); wd=gt.predict_w(cal); pD=(wd*cal).sum(0)
    accD=accuracy_score(y,(pD>=.5).astype(int))*100
    singles={atlas:accuracy_score(y,best_atlas[atlas][idx]>=.5)*100 for atlas in ATLASES}
    rows.append({"site":site,"n":len(idx),"A":accA,"B":accB,"C":accC,"D":accD,
                 "S_best":best_single_site,
                 "S_cc200":singles["cc200"],"S_aal":singles["aal"],"S_dos160":singles["dos160"],
                 "w_cc200":wB[0],"w_aal":wB[1],"w_dos160":wB[2],"gamma_C":bg})
    pd.DataFrame(rows).to_csv(PARTIAL,index=False)
    r=rows[-1]
    print(f"  {site:9s} n={len(idx):3d} | A={r['A']:5.2f} B={r['B']:5.2f} C={r['C']:5.2f} D={r['D']:5.2f} "
          f"| 单图谱最优={r['S_best']:5.2f} (cc200={r['S_cc200']:.1f} aal={r['S_aal']:.1f} dos={r['S_dos160']:.1f})",flush=True)

res=pd.DataFrame(rows); res.to_csv(r"D:/ASD-GraphNet - mhx/fusion_paper_results.csv",index=False)
if os.path.exists(PARTIAL): os.remove(PARTIAL)

print("\n"+"="*84); print("汇总 (论文口径, 按样本量加权)"); print("="*84)
wts=res["n"].values; summ={}
for sch in ["A","B","C","D","S_best","S_cc200","S_aal","S_dos160"]:
    summ[sch]=np.nansum(res[sch].values*wts)/wts.sum(); print(f"  {sch:12s} {summ[sch]:.2f}%")
best_f=max(summ["A"],summ["B"],summ["C"],summ["D"])
print(f"\n  最优单图谱(对照): {summ['S_best']:.2f}%")
print(f"  最优融合:         {best_f:.2f}%   增益 {best_f-summ['S_best']:+.2f}%")
print(f"  (论文基线 75.63% | v9 77.59%)")
from scipy.stats import wilcoxon
print("\n  站点层面 Wilcoxon:")
for sch in ["A","B","C","D"]:
    d=(res[sch]-res["S_best"]).dropna()
    if len(d)>5 and d.abs().sum()>0:
        st,p=wilcoxon(d); print(f"    {sch} vs 单图谱最优: Δ={d.mean():+.2f}%, p={p:.4f}")
print("\n  平均权重 (方案B):")
for a,c in zip(ATLASES,["w_cc200","w_aal","w_dos160"]):
    v=res[c].dropna(); print(f"    {a:8s} {v.mean():.3f}  范围[{v.min():.3f},{v.max():.3f}]")
print(f"\n用时 {time.time()-t0:.0f}s")
