"""05b: compare scalers with the original LORO protocol; resume completed combinations."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.metrics import accuracy_score, f1_score, recall_score

METHODS = {'N0':'None', 'N1':'StandardScaler', 'N2':'RobustScaler', 'N3':'MinMaxScaler', 'N4':'PowerTransformer(Yeo-Johnson)', 'N5':'Per-Trip Robust'}
SCALERS = {'N0':None,'N1':StandardScaler,'N2':RobustScaler,'N3':MinMaxScaler,'N4':PowerTransformer}
METRICS = ['視窗準確率','視窗MacroF1','捷運召回','公車召回','汽車召回','趟級準確率']
SUFFIX = {'RandomForest':'RF','SVM(RBF)':'SVM','kNN(k=15)':'kNN'}

def save(df, path):
    tmp = path.with_suffix('.tmp')
    df.to_csv(tmp,index=False,encoding='utf-8-sig'); tmp.replace(path)

def build_model(ev, name, method):
    original = ev.models()[name]
    estimator = original.steps[-1][1] if hasattr(original,'steps') else original
    scaler = SCALERS[method]
    return estimator if scaler is None else make_pipeline(scaler(),estimator)

def evaluate(ev, X, y, g, name, method):
    # Same RNG, fold ordering and training cap as 03. RF is never subsampled,
    # including when wrapped in a Pipeline (original loro checks estimator type).
    from sklearn.base import clone
    from sklearn.model_selection import LeaveOneGroupOut
    rng = np.random.default_rng(0)
    pred = np.empty(len(y),dtype=object)
    model = build_model(ev,name,method)
    n = len(np.unique(g))
    for fold,(tr,te) in enumerate(LeaveOneGroupOut().split(X,y,g),1):
        assert not set(g[tr]) & set(g[te])
        if name != 'RandomForest' and len(tr)>ev.MAX_TRAIN:
            tr = rng.choice(tr,ev.MAX_TRAIN,replace=False)
        pred[te] = clone(model).fit(X[tr],y[tr]).predict(X[te])
        if fold==1 or fold%10==0 or fold==n:
            print(f'  LORO {fold}/{n}',flush=True)
    return pred

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=list(SCALERS),help='Only run this method; default all')
    parser.add_argument('--model',choices=list(SUFFIX.values()),help='Only run this model; default all')
    args=parser.parse_args()
    ev=importlib.import_module('03_train_evaluate')
    root=Path(ev.OUT_DIR); out=root/'05b_scaler_comparison'; out.mkdir(parents=True,exist_ok=True)
    feature_path=root/'features.csv'; D=pd.read_csv(feature_path)
    cols=[c for c in D if c.startswith(('acc_','gyr_'))]
    X=D[cols].to_numpy(); y=D.cls.to_numpy(); g=D.trip.to_numpy()
    if len(cols)!=28 or not np.isfinite(X).all() or D[['trip','cls']].isna().any().any() or D.groupby('trip').cls.nunique().max()!=1 or set(y)!=set(ev.LABELS) or D.groupby('cls').trip.nunique().min()<2:
        raise ValueError('Invalid feature data or trip labels')
    a_path=root/'結果總表.csv'; b_path=root/'結果_A標準化.csv'
    A=pd.read_csv(a_path); A=A[A['實驗'].str.startswith('A ')]; B=pd.read_csv(b_path)
    for table in (A,B):
        if len(table)!=3 or set(table['模型'])!=set(SUFFIX) or not table['趟數'].eq(D.trip.nunique()).all() or not np.isfinite(table[METRICS].to_numpy()).all():
            raise ValueError('Prior results do not match current trip count/models')
        if '視窗數' in table and not table['視窗數'].eq(len(D)).all():
            raise ValueError('Prior window count mismatch')
    signature={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (feature_path,a_path,b_path,Path(ev.__file__),Path(__file__))}
    manifest=out/'run_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=signature:
        raise ValueError('Inputs/code changed; rename old 05b output folder before rerunning')
    manifest.write_text(json.dumps(signature,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for table,normalized in [(A,False),(B,True)]:
        for _,r in table.iterrows():
            name=r['模型']; method='N5' if normalized else ('N0' if name=='RandomForest' else 'N1')
            rows.append(dict(r,實驗=method,方法=METHODS[method]+(' + StandardScaler' if normalized and name!='RandomForest' else ''),來源='既有結果',視窗數=len(D)))
    for method in SCALERS:
        for name in SUFFIX:
            if (method=='N0' and name=='RandomForest') or (method=='N1' and name!='RandomForest'): continue
            path=out/f'{method}_{SUFFIX[name]}_metrics.csv'
            if path.exists():
                rows.extend(pd.read_csv(path).to_dict('records')); continue
            if args.method and args.method!=method or args.model and args.model!=SUFFIX[name]: continue
            print(f'Start {method} {METHODS[method]} / {name}',flush=True)
            p=evaluate(ev,X,y,g,name,method)
            T=pd.DataFrame(dict(trip=g,y=y,p=p)); T['ok']=y==p
            tv=T.groupby('trip').agg(y=('y','first'),p=('p',lambda s:s.value_counts().index[0]),視窗準確率=('ok','mean'),視窗數=('ok','size')).reset_index()
            rec=recall_score(y,p,labels=ev.LABELS,average=None,zero_division=0)
            row=dict(實驗=method,方法=METHODS[method],模型=name,來源='新增實驗',視窗準確率=round(accuracy_score(y,p),3),視窗MacroF1=round(f1_score(y,p,labels=ev.LABELS,average='macro'),3),捷運召回=round(rec[0],3),公車召回=round(rec[1],3),汽車召回=round(rec[2],3),趟級準確率=round((tv.y==tv.p).mean(),3),趟數=len(tv),視窗數=len(D))
            prefix=f'{method}_{SUFFIX[name]}'
            save(tv,out/f'{prefix}_每趟結果.csv')
            ev.plot_cm(y,p,f'{method} {METHODS[method]} / {name}',out/f'{prefix}_混淆矩陣.png')
            save(pd.DataFrame([row]),path); rows.append(row); print(row,flush=True)
    R=pd.DataFrame(rows).sort_values(['實驗','模型'])
    save(R,out/'標準化方法比較.csv')
    base=R[((R['實驗']=='N0')&(R['模型']=='RandomForest'))|((R['實驗']=='N1')&(R['模型']!='RandomForest'))]
    C=R.merge(base[['模型']+METRICS],on='模型',suffixes=('','_基準'),validate='many_to_one')
    for m in METRICS: C[m+'_差異百分點']=((C[m]-C[m+'_基準'])*100).round(1)
    save(C,out/'與實驗A比較.csv')
    for m in ['視窗準確率','視窗MacroF1','趟級準確率']:
        save(R.sort_values(m,ascending=False),out/f'排名_{m}.csv')
    print(R[['實驗','方法','模型']+METRICS].to_string(index=False)); print(f'Completed {len(R)}/18 combinations: {out}')

if __name__=='__main__': main()
