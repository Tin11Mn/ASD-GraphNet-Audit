"""
最终整合: 可靠性感知多图谱融合 接入完整配置池
目标: 融合模块能否突破 v9 的 77.59%

配置池 (每站点):
  - 单图谱配置: 3 atlas × {node_based, graph_level} × {mi_10, pca_10} = 12
  - 跨图谱拼接配置: 6 个 (tri_atlas / dual_atlas / full)
  模型: LR / RF / XGB
每个配置用 10-fold CV 得到 OOF 概率 (每被试一个预测值)

对照与融合:
  S_any  : 全池最优单个配置 (v9 口径的上界)
  S_3atlas: 仅单图谱配置中的最优
  F_top1 : 每图谱取 top-1 配置 -> 可靠性加权融合
  F_top2 : 每图谱取 top-2 配置 -> 图谱内平均 -> 可靠性加权融合
  F_eq   : top-1 等权融合
冲突分析: 三图谱预测不一致的子集上, 各方法表现
"""
import ast, warnings, time, os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
np.random.seed(42); t0=time.time()

DATA=r"D:/ASD-GraphNet - mhx/ASD-GraphNet-main/outputs/filtered_subjects_df.csv"
df=pd.read_csv(DATA); df["features_value"]=df["features_value"].apply(ast.literal_eval)
ATLASES=["cc200","aal","dos160"]; LVLS=["node_based","graph_level"]; FES=["mi_10","pca_10"]

X_by={}
for a in ATLASES:
    for l in LVLS:
        for f in FES:
            s=df[(df["atlas"]==a)&(df["graph_feature_type"]==l)&(df["feature_engineering"]==f)]
            if len(s)==0: continue
            d={int(r["id"]):np.array(r["features_value"],dtype=float) for _,r in s.iterrows()}
            if len({len(v) for v in d.values()})!=1: continue
            X_by[(a,l,f)]=d
# 跨图谱拼接配置
for a in ATLASES:
    for l in LVLS:
        for f in FES:
            parts=[X_by.get((x,l,f)) for x in ATLASES]
            if any(p is None for p in parts): continue
            ids=sorted(set.intersection(*[set(p.keys()) for p in parts]))
            X_by[("ALL",a,l,f)]={i:np.concatenate([p[i] for p in parts]) for i in ids}
for a in ATLASES:
    for l in LVLS:
        for f in FES:
            duo=[x for x in ATLASES if x!=a]
            if all((x,l,f) in X_by for x in duo):
                parts=[X_by[(x,l,f)] for x in duo]
                ids=sorted(set.intersection(*[set(p.keys()) for p in parts]))
                X_by[("DUAL",a,l,f)]={i:np.concatenate([p[i] for p in parts]) for i in ids}

meta=df[["id","ASD","SITE_ID"]].drop_duplicates(subset=["id"]).copy()
meta["id"]=meta["id"].astype(int); meta=meta.sort_values("id").reset_index(drop=True)
IDS=meta["id"].tolist(); Y=meta["ASD"].values; SITE=meta["SITE_ID"].values

def mk(n,seed):
    if n=="LR": return LogisticRegression(max_iter=1000,C=0.1)
    if n=="RF": return RandomForestClassifier(n_estimators=100,random_state=seed,n_jobs=1)
    return XGBClassifier(n_estimators=100,random_state=seed,eval_metric="logloss",verbosity=0,tree_method="hist")
MODELS=["LR","RF","XGB"]
EPS=1e-6
def to_l(p): p=np.clip(p,EPS,1-EPS); return np.log(p/(1-p)).reshape(-1,1)
def platt(o,y):
    if len(np.unique(y))<2: return None
    lr=LogisticRegression(C=1e4,max_iter=1000); lr.fit(to_l(o),y); return lr
def plapp(c,p): return p if c is None else c.predict_proba(to_l(p))[:,1]
def sm(pr): e=np.exp(pr-pr.max()); return e/e.sum()
def fit_w(op,y):
    def obj(pr):
        w=sm(pr); p=np.clip((w[:,None]*op).sum(0),EPS,1-EPS)
        return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
    best,bv=np.zeros(op.shape[0]),np.inf
    for init in [np.zeros(op.shape[0]),np.array([.5,-.5,0.]),np.array([-.5,.5,0.])]:
        if len(init)!=op.shape[0]: continue
        r=minimize(obj,init,method="Nelder-Mead",options={"maxiter":300,"xatol":1e-4,"fatol":1e-6})
        if r.fun<bv: bv,best=r.fun,sm(r.x)
    return best
def rel_w(op,y):
    nll=[]
    for a in range(op.shape[0]):
        p=np.clip(op[a],EPS,1-EPS); nll.append(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
    return sm(-np.array(nll)/max(np.std(nll),1e-6))

print(f"配置池: {len(X_by)} 个配置 × {len(MODELS)} 模型")
OUTER=10; SEED=42; rows=[]; PART=r"D:/ASD-GraphNet - mhx/fusion_int_partial.csv"
done=set()
if os.path.exists(PART):
    pv=pd.read_csv(PART); rows=pv.to_dict("records"); done=set(pv["site"].unique()); print(f"续跑 {sorted(done)}")

for site in sorted(set(SITE)):
    if site in done: continue
    idx=np.where(SITE==site)[0]; y=Y[idx]
    if len(idx)<25 or np.bincount(y,minlength=2).min()<6: continue
    oof_store={}
    skf=StratifiedKFold(OUTER,shuffle=True,random_state=SEED)
    for cfg,d in X_by.items():
        for mn in MODELS:
            try: A_all=np.stack([d[i] for i in IDS])
            except KeyError: continue
            oof=np.zeros(len(IDS))
            for tr,te in skf.split(A_all[idx],y):
                trg=idx[tr]; teg=idx[te]
                sc=StandardScaler(); A=sc.fit_transform(A_all[trg]); B=sc.transform(A_all[teg])
                m=mk(mn,SEED); m.fit(A,y[tr]); oof[teg]=m.predict_proba(B)[:,1]
            oof_store[(cfg,mn)]=oof
    accs={k:accuracy_score(y,(v[idx]>=.5).astype(int))*100 for k,v in oof_store.items()}
    # S_any / S_3atlas
    key_any=max(accs,key=accs.get)
    single_keys={k for k in oof_store if k[0][0] in ATLASES}
    key_3=max(single_keys,key=lambda k:accs[k]) if single_keys else key_any
    # 每图谱候选 (仅该图谱的纯配置)
    per_atlas={}
    for a in ATLASES:
        cands=[(k,v) for k,v in oof_store.items() if k[0][0]==a]
        cands.sort(key=lambda kv:-accs[kv[0]])
        per_atlas[a]=cands
    def fuse(topk,use_rel):
        ops=[];ts=[];yy=[]
        for a in ATLASES:
            if not per_atlas[a]: return None,None
            sel=per_atlas[a][:topk]
            op=np.mean([v[idx] for _,v in sel],axis=0)
            c=platt(op,y); ops.append(plapp(c,op)); ts.append(accs[sel[0][0]])
        ops=np.array(ops)
        w=rel_w(ops,y) if use_rel else np.ones(3)/3
        p=(w[:,None]*ops).sum(0)
        return accuracy_score(y,(p>=.5).astype(int))*100, ops
    accF1,ops1=fuse(1,True); accF2,ops2=fuse(2,True); accEq,_=fuse(1,False)
    # 冲突分析 (基于 top-1 校准概率)
    conflict_n=0; conflict_fuse_acc=None; conflict_single=[]
    if ops1 is not None:
        preds=(ops1>=.5).astype(int)
        disagree=(preds!=preds[0]).any(axis=0)
        conflict_n=int(disagree.sum())
        if conflict_n>0:
            w=rel_w(ops1,y); pf=(w[:,None]*ops1).sum(0)
            conflict_fuse_acc=accuracy_score(y[disagree],(pf[disagree]>=.5).astype(int))*100
            conflict_single=[accuracy_score(y[disagree],(ops1[i][disagree]>=.5).astype(int))*100 for i in range(3)]
    rows.append({"site":site,"n":len(idx),"S_any":accs[key_any],"S_3atlas":accs[key_3],
                 "F_top1":accF1,"F_top2":accF2,"F_eq":accEq,
                 "best_cfg":str(key_any),"conflict_n":conflict_n,
                 "conflict_fuse":conflict_fuse_acc,
                 "c_cc200":conflict_single[0] if conflict_single else np.nan,
                 "c_aal":conflict_single[1] if conflict_single else np.nan,
                 "c_dos160":conflict_single[2] if conflict_single else np.nan})
    pd.DataFrame(rows).to_csv(PART,index=False)
    r=rows[-1]
    print(f"  {site:9s} n={len(idx):3d} | S_any={r['S_any']:5.2f} S_3atlas={r['S_3atlas']:5.2f} "
          f"| F_top1={r['F_top1']:5.2f} F_top2={r['F_top2']:5.2f} F_eq={r['F_eq']:5.2f} "
          f"| 冲突={r['conflict_n']:3d}",flush=True)

res=pd.DataFrame(rows); res.to_csv(r"D:/ASD-GraphNet - mhx/fusion_integrated.csv",index=False)
if os.path.exists(PART): os.remove(PART)

print("\n"+"="*84); print("汇总 (论文口径, 按样本量加权)"); print("="*84)
wts=res["n"].values; S={}
for c in ["S_any","S_3atlas","F_top1","F_top2","F_eq"]:
    S[c]=np.nansum(res[c].values*wts)/wts.sum(); print(f"  {c:10s} {S[c]:.2f}%")
print(f"\n  v9 基准: 77.59% | 论文基线: 75.63%")
print(f"  融合 vs 全池最优单配置: {S['F_top1']-S['S_any']:+.2f}%")
print(f"  融合 vs 最优单图谱:     {S['F_top1']-S['S_3atlas']:+.2f}%")
from scipy.stats import wilcoxon
for c in ["F_top1","F_top2","F_eq"]:
    d=(res[c]-res["S_3atlas"]).dropna()
    if len(d)>5 and d.abs().sum()>0:
        st,p=wilcoxon(d); print(f"  Wilcoxon {c} vs S_3atlas: Δ={d.mean():+.2f}%, p={p:.4f}")
# 冲突分析汇总
cn=res["conflict_n"].sum()
if cn>0:
    cw=res["n"]/res["n"].sum()
    print(f"\n  冲突样本分析 (三图谱预测不一致, 共 {cn} 个):")
    print(f"    融合方法准确率: {np.nansum(res['conflict_fuse']*cw):.2f}%")
    for c,nm in [("c_cc200","cc200"),("c_aal","aal"),("c_dos160","dos160")]:
        print(f"    {nm:8s}单独: {np.nansum(res[c]*cw):.2f}%")
print(f"\n用时 {time.time()-t0:.0f}s")
