"""06: phone-position experiments; RF raw, RF per-trip Robust and SVM Yeo-Johnson. Run from project root."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import sklearn

POSITIONS = ['hand', 'pocket', 'bag']
MODELS = {'RF': ('RandomForest', 'N0'), 'RF_N5': ('RandomForest', 'N5'), 'SVM': ('SVM(RBF)', 'N4')}
METRICS = ['視窗準確率','視窗MacroF1','趟級準確率','趟級MacroF1','捷運召回','公車召回','汽車召回']

def save(df, path):
    tmp = path.with_suffix('.tmp'); df.to_csv(tmp,index=False,encoding='utf-8-sig'); tmp.replace(path)

def evaluate(D, cols, model, key, ev, mode, path):
    """Cache complete out-of-fold predictions; no test windows used in model fit. N5 uses whole-trip offline normalization."""
    X=D[cols].to_numpy(); y=D.cls.to_numpy(); g=D.trip.to_numpy()
    if path.exists():
        old=pd.read_csv(path)
        if not old[['row_id','trip','cls','position']].equals(D[['row_id','trip','cls','position']].reset_index(drop=True)) or not old.pred.isin(ev.LABELS).all():
            raise ValueError(f'Invalid cache: {path}')
        return old
    if mode=='E2':
        splits=[(np.flatnonzero(D.position.to_numpy()!=p),np.flatnonzero(D.position.to_numpy()==p)) for p in POSITIONS]
    else:
        splits=list(LeaveOneGroupOut().split(X,y,g))
    rng=np.random.default_rng(0); pred=np.full(len(D),None,dtype=object); seen=np.zeros(len(D),int)
    for i,(tr,te) in enumerate(splits,1):
        if set(g[tr]) & set(g[te]): raise ValueError('Trip leakage')
        if key=='SVM' and len(tr)>ev.MAX_TRAIN: tr=rng.choice(tr,ev.MAX_TRAIN,replace=False)
        if set(y[tr])!=set(ev.LABELS): raise ValueError('Training fold lacks a class')
        pred[te]=clone(model).fit(X[tr],y[tr]).predict(X[te]); seen[te]+=1
        if i==1 or i%10==0 or i==len(splits): print(f'  {path.stem}: {i}/{len(splits)}',flush=True)
    if not np.all(seen==1): raise ValueError('Incomplete or duplicate predictions')
    T=D[['row_id','trip','cls','position','device']].copy(); T['pred']=pred
    save(T,path); return T

def trips(T):
    T=T.assign(ok=T.cls==T.pred)
    return T.groupby('trip').agg(cls=('cls','first'),position=('position','first'),device=('device','first'),pred=('pred',lambda s:s.value_counts().index[0]),視窗準確率=('ok','mean'),視窗數=('ok','size')).reset_index()

def summarize(T, ev, key, exp, pos):
    y=T.cls; p=T.pred; V=trips(T); labels=ev.LABELS
    if set(y)!=set(labels): raise ValueError(f'Missing test class: {exp}/{pos}')
    rec=recall_score(y,p,labels=labels,average=None,zero_division=0)
    return dict(模型=key,實驗=exp,位置=pos,趟數=len(V),視窗數=len(T),視窗準確率=accuracy_score(y,p),視窗MacroF1=f1_score(y,p,labels=labels,average='macro',zero_division=0),趟級準確率=accuracy_score(V.cls,V.pred),趟級MacroF1=f1_score(V.cls,V.pred,labels=labels,average='macro',zero_division=0),捷運召回=rec[0],公車召回=rec[1],汽車召回=rec[2])

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--model',choices=MODELS); args=ap.parse_args()
    ev=importlib.import_module('03_train_evaluate'); sc=importlib.import_module('05b_scaler_comparison')
    root=Path(ev.OUT_DIR); source=root/'features.csv'; out=root/'06_core_experiments'; out.mkdir(parents=True,exist_ok=True)
    D=pd.read_csv(source); D['row_id']=np.arange(len(D)); cols=[c for c in D if c.startswith(('acc_','gyr_'))]
    meta=['trip','cls','position','device']
    if len(cols)!=28 or not np.isfinite(D[cols].to_numpy()).all() or D[meta].isna().any().any() or (D.groupby('trip')[meta[1:]].nunique()>1).any().any() or set(D.position)!=set(POSITIONS) or set(D.cls)!=set(ev.LABELS): raise ValueError('Invalid features or trip metadata')
    counts=D.groupby(['position','cls']).trip.nunique().unstack(fill_value=0).reindex(index=POSITIONS,columns=ev.LABELS,fill_value=0)
    if (counts<2).any().any(): raise ValueError('Each position/class needs at least two trips for LORO')
    paths=[source,Path(__file__),Path(ev.__file__),Path(sc.__file__),Path(importlib.import_module('common').__file__)]
    sig={'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'versions':{'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__}}
    manifest=out/'run_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=sig: raise ValueError('Inputs/code changed; rename old output directory')
    manifest.write_text(json.dumps(sig,ensure_ascii=False,indent=2),encoding='utf-8')
    save(D.groupby(['position','cls']).agg(趟數=('trip','nunique'),視窗數=('trip','size')).reset_index(),out/'資料分布.csv')
    # N5 uses each complete trip independently, including held-out trips.
    # This is offline inference, not a training-fold global scaler.
    Z=importlib.import_module('common').normalize_per_trip(D,cols)
    if not np.isfinite(Z[cols].to_numpy()).all(): raise ValueError('Non-finite N5 features')
    rows=[]; details=[]; cms=[]
    for key,(name,method) in MODELS.items():
        model=sc.build_model(ev,name,'N0' if method=='N5' else method)
        F=Z if method=='N5' else D
        jobs=[('E1','all',F),('E2','all',F)]+[('E3',p,F[F.position==p].reset_index(drop=True)) for p in POSITIONS]
        for exp,pos,sub in jobs:
            path=out/f'{key}_{exp}_{pos}_predictions.csv'
            if not path.exists() and args.model and args.model!=key: continue
            T=evaluate(sub,cols,model,key,ev,exp,path)
            V=trips(T); save(V,out/f'{key}_{exp}_{pos}_每趟結果.csv')
            scopes=[('all',T)]+[(p,T[T.position==p]) for p in POSITIONS] if pos=='all' else [(pos,T)]
            for p,U in scopes:
                rows.append(dict(summarize(U,ev,key,exp,p),分類器=name,方法=sc.METHODS[method]))
                for cls in ev.LABELS:
                    W=U[U.cls==cls]; details.append(dict(模型=key,實驗=exp,位置=p,類別=cls,趟數=W.trip.nunique(),視窗數=len(W),召回率=(W.cls==W.pred).mean()))
                for level,A,B in [('window',U.cls,U.pred),('trip',trips(U).cls,trips(U).pred)]:
                    cm=confusion_matrix(A,B,labels=ev.LABELS)
                    for i,a in enumerate(ev.LABELS):
                        for j,b in enumerate(ev.LABELS): cms.append(dict(模型=key,實驗=exp,位置=p,層級=level,實際=a,預測=b,數量=cm[i,j]))
                ev.plot_cm(U.cls.to_numpy(),U.pred.to_numpy(),f'{key} {exp} {p}',out/f'{key}_{exp}_{p}_混淆矩陣.png')
    R=pd.DataFrame(rows); save(R,out/'結果_核心實驗.csv'); save(pd.DataFrame(details),out/'各位置類別召回.csv'); save(pd.DataFrame(cms),out/'混淆矩陣數值.csv')
    base=R[(R.實驗=='E1')&(R.位置!='all')]; other=R[(R.實驗!='E1')&(R.位置!='all')]
    C=other.merge(base[['模型','位置','趟數','視窗數']+METRICS],on=['模型','位置'],suffixes=('','_統一'),validate='many_to_one')
    if not C.empty:
        for c in ['趟數','視窗數']:
            if not C[c].eq(C[c+'_統一']).all(): raise ValueError('Comparison test counts mismatch')
        for m in METRICS: C[m+'_差異百分點']=(C[m]-C[m+'_統一'])*100
    save(C,out/'與統一模型比較.csv')
    # Compare the two RF variants on identical evaluation scopes.
    raw=R[R.模型=='RF']; norm=R[R.模型=='RF_N5']
    Q=norm.merge(raw[['實驗','位置','趟數','視窗數']+METRICS],on=['實驗','位置'],suffixes=('_每趟標準化','_不標準化'),validate='one_to_one')
    for c in ['趟數','視窗數']:
        if not Q[c+'_每趟標準化'].eq(Q[c+'_不標準化']).all(): raise ValueError('RF comparison test counts mismatch')
    for m in METRICS: Q[m+'_差異百分點']=(Q[m+'_每趟標準化']-Q[m+'_不標準化'])*100
    save(Q,out/'RF_每趟標準化比較.csv')
    print(R.round(3).to_string(index=False)); print(f'Completed {len(R)}/33 summary rows. {out}')

if __name__=='__main__': main()
