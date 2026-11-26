"""步驟5：每趟錄製 Median/IQR 標準化；與實驗A比較（離線辨識）。"""
import importlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from common import OUT_DIR, CLASS_NAMES, normalize_per_trip

ev = importlib.import_module('03_train_evaluate')

def save(df, name):
    df.to_csv(OUT_DIR / name, index=False, encoding='utf-8-sig')

def main():
    X = pd.read_csv(OUT_DIR / 'features.csv')
    cols = [c for c in X if c.startswith(('acc_', 'gyr_'))]
    if len(cols) != 28 or not np.isfinite(X[cols].to_numpy()).all():
        raise ValueError('預期28個有限數值特徵')
    if X[['trip','cls','device','position']].isna().any().any() or X.groupby('trip').cls.nunique().max()!=1:
        raise ValueError('錄製標籤異常')
    if set(X.cls)!=set(ev.LABELS) or X.groupby('cls').trip.nunique().min()<2:
        raise ValueError('類別或錄製數不足')
    baseline = pd.read_csv(OUT_DIR / '結果總表.csv')
    baseline = baseline[baseline['實驗'].str.startswith('A ')].copy()
    if set(baseline['模型'])!=set(ev.models()) or len(baseline)!=3 or not baseline['趟數'].eq(X.trip.nunique()).all():
        raise ValueError('實驗A結果不符，請確認資料版本')
    Z = normalize_per_trip(X, cols)
    if not np.isfinite(Z[cols].to_numpy()).all():
        raise ValueError('標準化後出現非有限值')
    print(f'標準化完成：{len(Z)} 視窗，{Z.trip.nunique()} 筆錄製', flush=True)
    y, g = Z.cls.to_numpy(), Z.trip.to_numpy()
    rows, devices, positions, trips = [], [], [], []
    metrics = ['視窗準確率','視窗MacroF1','捷運召回','公車召回','汽車召回','趟級準確率']
    for name, model in ev.models().items():
        print(f"開始 A標準化 / {name}", flush=True)
        p = ev.loro(Z[cols].to_numpy(), y, g, model)
        T = X[['trip','cls','device','position']].copy()
        T['pred'] = p; T['ok'] = y == p
        T['平台'] = np.where(T.device.str.contains('iPhone',case=False),'iPhone','Android')
        tv = T.groupby('trip').agg(y=('cls','first'), p=('pred',lambda s:s.value_counts().index[0]), 手機=('device','first'),位置=('position','first'),視窗準確率=('ok','mean'),視窗數=('ok','size')).reset_index()
        tv['模型']=name; tv['趟級正確']=tv.y==tv.p; trips.append(tv)
        rec = recall_score(y,p,labels=ev.LABELS,average=None,zero_division=0)
        row = dict(實驗="A' 全部資料(每趟標準化)",模型=name,視窗準確率=round(accuracy_score(y,p),3),視窗MacroF1=round(f1_score(y,p,labels=ev.LABELS,average='macro'),3),趟級準確率=round(tv['趟級正確'].mean(),3),趟數=len(tv),視窗數=len(T))
        row.update({CLASS_NAMES[c]+'召回':round(v,3) for c,v in zip(ev.LABELS,rec)})
        rows.append(row); print(row,flush=True)
        save(pd.DataFrame(rows),'結果暫存_A標準化.csv')
        suffix={'RandomForest':'RF','SVM(RBF)':'SVM','kNN(k=15)':'kNN'}[name]
        ev.plot_cm(y,p,f"實驗A'（每趟標準化）{name}",OUT_DIR / f'混淆矩陣_A標準化_{suffix}.png')
        for keys, target in [(['cls','平台'],devices),(['cls','position'],positions)]:
            d=T.groupby(keys).ok.agg(視窗準確率='mean',視窗數='size').reset_index(); d['模型']=name; target.append(d)
    R=pd.DataFrame(rows); save(R,'結果_A標準化.csv')
    comparison=baseline[['模型']+metrics].merge(R[['模型']+metrics],on='模型',suffixes=('_前','_後'),validate='one_to_one')
    for m in metrics:
        comparison[m+'_差異百分點']=((comparison[m+'_後']-comparison[m+'_前'])*100).round(1)
    save(comparison,'標準化前後比較.csv')
    save(pd.concat(devices,ignore_index=True).round(3),'裝置診斷_A標準化.csv')
    save(pd.concat(positions,ignore_index=True).round(3),'位置影響_A標準化.csv')
    save(pd.concat(trips,ignore_index=True).round(3),'每趟結果_A標準化.csv')
    print(comparison.to_string(index=False)); print('第五階段完成：',OUT_DIR)

if __name__=='__main__':
    main()
