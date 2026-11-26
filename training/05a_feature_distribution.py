"""05a: descriptive feature diagnostics; no model fitting or input modification."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = ['MRT', 'bus', 'car']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.input is None:
        from common import OUT_DIR
        args.input = OUT_DIR / 'features.csv'
    out = args.output or args.input.parent / '05a_feature_distribution'
    out.mkdir(parents=True, exist_ok=True)
    D = pd.read_csv(args.input)
    cols = [c for c in D if c.startswith(('acc_', 'gyr_'))]
    if len(cols) != 28 or not np.isfinite(D[cols].to_numpy(dtype=float)).all():
        raise ValueError('預期28個有限數值特徵')
    if D[['trip','cls']].isna().any().any() or set(D.cls) != set(LABELS) or D.groupby('trip').cls.nunique().max()!=1:
        raise ValueError('錄製或類別標籤異常')
    X = D[cols]
    def save(df, name, index=False):
        df.to_csv(out / name, index=index, encoding='utf-8-sig')
    def stats(x):
        q1, med, q3 = x.quantile([.25,.5,.75]); iqr=q3-q1
        return dict(n=len(x), mean=x.mean(), std=x.std(), min=x.min(), p01=x.quantile(.01), q1=q1, median=med, q3=q3, p99=x.quantile(.99), max=x.max(), iqr=iqr, skew=x.skew(), kurtosis=x.kurt(), zero_pct=100*x.eq(0).mean(), tukey_outlier_pct=100*((x<q1-1.5*iqr)|(x>q3+1.5*iqr)).mean(), constant=x.nunique()==1, zero_iqr=iqr==0)
    rows=[]; groups=[]; changes=[]
    for c in cols:
        x=X[c]; s=stats(x)
        rows.append(dict(feature=c, **s, strong_skew=abs(s['skew'])>1, outlier_over_5pct=s['tukey_outlier_pct']>5))
        for label in LABELS:
            groups.append(dict(feature=c, cls=label, **stats(D.loc[D.cls==label,c])))
        g=D.groupby('trip')[c]; med=g.transform('median'); iqr=g.transform(lambda v:v.quantile(.75)-v.quantile(.25))
        z=(x-med)/(iqr+1e-9)
        # Descriptive between-class variance fraction, not a validation score.
        def eta(v):
            total=((v-v.mean())**2).sum()
            return sum(len(a)*(a.mean()-v.mean())**2 for _,a in v.groupby(D.cls))/total if total>0 else np.nan
        changes.append(dict(feature=c, eta_squared_raw=eta(x), eta_squared_per_trip=eta(z), zero_iqr_trips=int(g.agg(lambda v:v.quantile(.75)-v.quantile(.25)).eq(0).sum()), normalized_abs_max=z.abs().max()))
    S=pd.DataFrame(rows); save(S,'特徵統計.csv'); save(pd.DataFrame(groups),'各類別特徵統計.csv'); save(pd.DataFrame(changes),'每趟標準化特徵比較.csv')
    save(D.groupby('cls').agg(windows=('trip','size'),recordings=('trip','nunique')).reset_index(),'資料分布.csv')
    for method in ['pearson','spearman']:
        C=X.corr(method=method); save(C,f'相關矩陣_{method}.csv',True)
        pairs=[dict(feature_1=a,feature_2=b,correlation=C.loc[a,b]) for i,a in enumerate(cols) for b in cols[i+1:] if abs(C.loc[a,b])>=.9]
        P=pd.DataFrame(pairs,columns=['feature_1','feature_2','correlation']); save(P.sort_values('correlation',key=lambda v:v.abs(),ascending=False),f'高度相關_{method}.csv')
        fig,ax=plt.subplots(figsize=(12,10)); im=ax.imshow(C,vmin=-1,vmax=1,cmap='coolwarm'); ax.set_xticks(range(len(cols)),cols,rotation=90,fontsize=8); ax.set_yticks(range(len(cols)),cols,fontsize=8); fig.colorbar(im,ax=ax); ax.set_title(method.title()+' correlation'); fig.tight_layout(); fig.savefig(out/f'correlation_{method}.png',dpi=150); plt.close(fig)
    for prefix in ['acc_','gyr_']:
        selected=[c for c in cols if c.startswith(prefix)]
        for kind in ['hist','box']:
            fig,axes=plt.subplots(4,4,figsize=(16,12)); axes=axes.ravel()
            for ax,c in zip(axes,selected):
                x=X[c]; lo,hi=x.quantile([.01,.99]); ax.set_title(c)
                if kind=='hist':
                    if lo==hi: lo,hi=x.min(),x.max()
                    if lo==hi: lo,hi=lo-1,hi+1
                    bins=np.linspace(lo,hi,41)
                    for label in LABELS:
                        v=D.loc[D.cls==label,c].to_numpy(); counts,_=np.histogram(v,bins=bins); ax.stairs(counts/len(v),bins,label=label)
                    ax.set_ylabel('Fraction of class / bin')
                else:
                    ax.boxplot([D.loc[D.cls==label,c] for label in LABELS],showfliers=False); ax.set_xticks([1,2,3],LABELS)
                ax.tick_params(labelsize=8)
            for ax in axes[len(selected):]: ax.axis('off')
            if kind=='hist': axes[0].legend(fontsize=8)
            fig.suptitle(f'{prefix[:-1]}: '+('Histograms (central 98% pooled range; fallback for ties)' if kind=='hist' else 'Class boxplots (fliers hidden)'))
            fig.tight_layout(); fig.savefig(out/f'{prefix}{kind}.png',dpi=150); plt.close(fig)
    print(f'有效視窗：{len(D)}；錄製：{D.trip.nunique()}；特徵：{len(cols)}')
    print(S[['feature','std','iqr','skew','tukey_outlier_pct','zero_iqr']].round(4).to_string(index=False))
    print('明顯偏態特徵：',S.loc[S.strong_skew,'feature'].tolist())
    print('Tukey界外值超過5%：',S.loc[S.outlier_over_5pct,'feature'].tolist())
    print('輸出：',out)

if __name__=='__main__': main()
