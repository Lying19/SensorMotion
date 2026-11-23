"""步驟 2：前處理 + 特徵擷取。
流程：讀 CSV -> 裁掉頭尾 -> 重新取樣到 50 Hz -> 算 Magnitude(與手機朝向無關) -> 2 秒視窗/50% 重疊
      -> 丟掉含資料中斷的視窗 -> 擷取時域與頻域特徵 -> 輸出 features.csv"""
import os, numpy as np, pandas as pd
from scipy import signal
from common import *

def mag_resampled(df):
    t = df["seconds_elapsed"].values
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

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, qual = [], []
    recs = find_recordings()
    for i, r in enumerate(recs, 1):
        a = pd.read_csv(os.path.join(r["dir"], "Accelerometer.csv"), usecols=["seconds_elapsed", "x", "y", "z"])
        g = pd.read_csv(os.path.join(r["dir"], "Gyroscope.csv"), usecols=["seconds_elapsed", "x", "y", "z"])
        md = pd.read_csv(os.path.join(r["dir"], "Metadata.csv")).iloc[0]
        ma, va = mag_resampled(a); mg, vg = mag_resampled(g)
        if ma is None or mg is None:
            print(f"[{i}/{len(recs)}] {r['trip']}: 太短，略過"); continue
        n = min(len(ma), len(mg)); ma, mg, v = ma[:n], mg[:n], va[:n] & vg[:n]
        total = ok = 0
        for s in range(0, n - WIN + 1, STEP):
            total += 1
            if not v[s:s+WIN].all(): continue
            ok += 1
            row = {}; row.update(feats(ma[s:s+WIN], "acc")); row.update(feats(mg[s:s+WIN], "gyr"))
            row.update(trip=r["trip"], cls=r["cls"], position=r["position"], device=md["device name"], date_tag=r["date_tag"])
            rows.append(row)
        qual.append(dict(trip=r["trip"], cls=r["cls"], position=r["position"], device=md["device name"],
                         total_windows=total, valid_windows=ok, valid_pct=round(100*ok/max(total, 1), 1)))
        print(f"[{i}/{len(recs)}] {r['trip']:20s} 有效視窗 {ok}/{total}")
    X = pd.DataFrame(rows); Q = pd.DataFrame(qual)
    X.to_csv(os.path.join(OUT_DIR, "features.csv"), index=False)
    Q.to_csv(os.path.join(OUT_DIR, "視窗品質.csv"), index=False, encoding="utf-8-sig")
    print("\n各類別有效視窗數："); print(X.groupby("cls").size())
