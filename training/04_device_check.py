"""步驟4：裝置診斷、實驗C（二分類）、實驗D（單一手機型號）。"""
import importlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from common import OUT_DIR, CLASS_NAMES

ev = importlib.import_module('03_train_evaluate')
plt = ev.plt

def save(df, name):
    df.to_csv(OUT_DIR / name, index=False, encoding='utf-8-sig')

def diagnose():
    p = pd.read_csv(OUT_DIR / '每趟結果_A.csv')
    p['平台'] = np.where(p['手機'].str.contains('iPhone', case=False, na=False), 'iPhone', 'Android')
    for key, filename in [('平台', '裝置診斷.csv'), ('手機', '裝置型號診斷.csv')]:
        rows = []
        for (cls, device), d in p.groupby(['類別', key]):
            rows.append({'類別': cls, key: device, '趟數': len(d), '視窗數': d['視窗數'].sum(),
                         '視窗準確率': round(np.average(d['準確率'], weights=d['視窗數']), 3)})
        result = pd.DataFrame(rows)
        save(result, filename)
        print(result.to_string(index=False))

def plot_cm(y, pred, labels, tag):
    cm = confusion_matrix(y, pred, labels=labels)
    rates = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(rates, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{rates[i,j]:.0%}\n({cm[i,j]})', ha='center', va='center', color='white' if rates[i,j] > .5 else 'black')
    zh = [CLASS_NAMES[c] for c in labels]
    ax.set_xticks(range(len(labels)), zh); ax.set_yticks(range(len(labels)), zh)
    ax.set_xlabel('預測'); ax.set_ylabel('實際'); ax.set_title(f'實驗{tag} RandomForest\nleave-one-recording-out')
    fig.tight_layout(); fig.savefig(OUT_DIR / f'混淆矩陣_{tag}.png', dpi=160); plt.close(fig)

def evaluate(d, cols, labels, tag, title):
    x, y, g = d[cols].to_numpy(), d.cls.to_numpy(), d.trip.to_numpy()
    if set(y) != set(labels) or d.groupby('cls').trip.nunique().min() < 2:
        raise ValueError(f'實驗{tag}類別或錄製數不足')
    print(f'實驗{tag}：{len(d)} 視窗，{d.trip.nunique()} 筆錄製', flush=True)
    rows = []
    for name, model in ev.models().items():
        print(f'開始 {tag} / {name}', flush=True)
        pred = ev.loro(x, y, g, model)
        rec = recall_score(y, pred, labels=labels, average=None, zero_division=0)
        votes = pd.DataFrame({'trip':g,'y':y,'p':pred}).groupby('trip').agg(y=('y','first'), p=('p',lambda s:s.value_counts().index[0]))
        row = {'實驗':title, '模型':name, '視窗準確率':round(accuracy_score(y,pred),3),
               '視窗MacroF1':round(f1_score(y,pred,labels=labels,average='macro',zero_division=0),3),
               '趟級準確率':round((votes.y==votes.p).mean(),3), '趟數':len(votes), '視窗數':len(d)}
        for label, value in zip(labels,rec):
            row[CLASS_NAMES[label]+'召回'] = round(value,3)
        rows.append(row); print(row, flush=True)
        save(pd.DataFrame(rows), f'結果暫存_{tag}.csv')
        if name == 'RandomForest':
            plot_cm(y,pred,labels,tag)
    result = pd.DataFrame(rows); save(result,f'結果_{tag}.csv'); return result

def main():
    x = pd.read_csv(OUT_DIR / 'features.csv')
    cols = [c for c in x if c.startswith(('acc_','gyr_'))]
    if len(cols)!=28 or not np.isfinite(x[cols].to_numpy()).all():
        raise ValueError('預期28個有限數值特徵')
    if x[['trip','cls','device']].isna().any().any() or x.groupby('trip').cls.nunique().max()!=1:
        raise ValueError('錄製標籤異常')
    print('手機 x 類別（錄製筆數）：'); print(x.groupby(['device','cls']).trip.nunique().unstack(fill_value=0))
    diagnose()
    c = x[x.device.str.contains('iPhone',case=False,na=False) & x.cls.isin(['bus','car'])]
    d = x[x.device.str.strip().eq('iPhone 17 Pro Max')]
    rc = evaluate(c,cols,['bus','car'],'C','C iPhone 公車vs汽車')
    rd = evaluate(d,cols,['MRT','bus','car'],'D','D iPhone 17 Pro Max')
    save(pd.concat([rc,rd],ignore_index=True),'結果_CD.csv')
    print('第四階段完成：',OUT_DIR)

if __name__ == '__main__':
    main()
