"""步驟 2：前處理 + 特徵擷取。
流程：讀 CSV -> 裁掉頭尾 -> 重新取樣到 50 Hz -> 算 Magnitude(與手機朝向無關) -> 2 秒視窗/50% 重疊
      -> 丟掉含資料中斷的視窗 -> 擷取時域與頻域特徵 -> 輸出 features.csv"""
import os, numpy as np, pandas as pd
from scipy import signal
from common import *

def mag_resampled(df):
    t = df["seconds_elapsed"].to_numpy(dtype=float)
    if len(t) < 2 or not np.isfinite(df.to_numpy(dtype=float)).all() or (np.diff(t) <= 0).any():
        raise ValueError("感測器數值或時間戳異常")
    m = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2).values
    grid = np.arange(t[0] + TRIM_SEC, t[-1] - TRIM_SEC, 1/FS)           # 裁掉頭尾
    if len(grid) < WIN: return None, None
    out = np.interp(grid, t, m)
    idx = np.searchsorted(t, grid).clip(1, len(t)-1)
    valid = (t[idx] - t[idx-1]) <= MAX_GAP                               # 落在大中斷內的點標為無效
    return out, valid

def feats(x, prefix):
    mean = x.mean(); xc = x - mean
    f, P = signal.welch(xc, FS, nperseg=len(xc)); Pn = P / (P.sum() + 1e-12)
    d = dict(mean=mean, std=xc.std(), max=x.max(), min=x.min(), rms=np.sqrt((xc**2).mean()),
             skew=pd.Series(xc).skew(), kurt=pd.Series(xc).kurt(), zcr=((xc[:-1]*xc[1:]) < 0).mean(),
             energy=(xc**2).mean(), domf=f[Pn.argmax()], spec_ent=-(Pn*np.log(Pn+1e-12)).sum(),
             p_low=Pn[f <= 3].sum(), p_mid=Pn[(f > 3) & (f <= 10)].sum(), p_high=Pn[f > 10].sum())
    return {f"{prefix}_{k}": v for k, v in d.items()}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, qual = [], []
    recs = find_recordings()
    if not recs:
        raise SystemExit("找不到原始資料")
    for i, r in enumerate(recs, 1):
        q = dict(trip=r['trip'], cls=r['cls'], position=r['position'], device='', total_windows=0, valid_windows=0, valid_pct=0, status='OK')
        local = []
        try:
            cols = ['seconds_elapsed', 'x', 'y', 'z']
            a = read_sensor(r, 'Accelerometer.csv', usecols=cols)
            g = read_sensor(r, 'Gyroscope.csv', usecols=cols)
            q['device'] = read_sensor(r, 'Metadata.csv').iloc[0]['device name']
            ma, va = mag_resampled(a); mg, vg = mag_resampled(g)
            if ma is None or mg is None:
                q['status'] = '裁切後太短'
            else:
                # 沿用同學的各感測器獨立取樣、依較短長度截齊方式。
                n = min(len(ma), len(mg))
                v = va[:n] & vg[:n]
                for start in range(0, n-WIN+1, STEP):
                    q['total_windows'] += 1
                    if not v[start:start+WIN].all(): continue
                    row = feats(ma[start:start+WIN], 'acc')
                    row.update(feats(mg[start:start+WIN], 'gyr'))
                    if not np.isfinite(list(row.values())).all(): continue
                    row.update(trip=r['trip'], cls=r['cls'], position=r['position'], device=q['device'], date_tag=r['date_tag'])
                    local.append(row)
                q['valid_windows'] = len(local)
                q['valid_pct'] = round(100*len(local)/max(q['total_windows'],1),1)
                if not local: q['status'] = '無有效視窗'
        except Exception as e:
            q['status'] = f'處理失敗: {e}'
            local = []
            q['valid_windows'] = q['valid_pct'] = 0
        rows.extend(local); qual.append(q)
        print(f"[{i}/{len(recs)}] {r['trip']} 有效視窗 {q['valid_windows']}/{q['total_windows']} {q['status']}", flush=True)
    Q = pd.DataFrame(qual)
    Q.to_csv(OUT_DIR / '視窗品質.csv', index=False, encoding='utf-8-sig')
    X = pd.DataFrame(rows, columns=list(feats(np.arange(WIN,dtype=float),'acc'))+list(feats(np.arange(WIN,dtype=float),'gyr'))+['trip','cls','position','device','date_tag'])
    X.to_csv(OUT_DIR / 'features.csv', index=False)
    if X.empty: raise SystemExit('無有效特徵，請檢查視窗品質.csv')
    print('各類別有效視窗數：'); print(X.groupby('cls').size())
    print('有效視窗總數：', len(X), '有效錄製筆數：', X.trip.nunique())
    print('輸出：', OUT_DIR / 'features.csv', OUT_DIR / '視窗品質.csv')

if __name__ == '__main__':
    main()
