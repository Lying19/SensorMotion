"""資料盤點：不修改原始資料。"""
import numpy as np
import pandas as pd
from common import DATA_DIR, OUT_DIR, find_recordings, read_sensor, has_sensor, MAX_GAP

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in find_recordings():
        row = dict(trip=r['trip'], 類別=r['cls'], 位置=r['position'], 來源=r['source'])
        try:
            row['有陀螺儀'] = has_sensor(r, 'Gyroscope.csv')
            a = read_sensor(r, 'Accelerometer.csv', usecols=['seconds_elapsed','x','y','z'])
            t = a['seconds_elapsed'].to_numpy(dtype=float)
            if len(t)<2 or not np.isfinite(t).all() or t[-1]<=t[0]:
                raise ValueError('時間戳不足或無效')
            dt = np.diff(t)
            row.update(分鐘=round((t[-1]-t[0])/60, 1), 實際取樣率Hz=round(len(t)/(t[-1]-t[0]),1),
                       中斷次數_大於1秒=int((dt>1).sum()), 中斷總秒數=int(dt[dt>1].sum()),
                       中斷次數_大於0點5秒=int((dt>MAX_GAP).sum()), 非遞增時間戳=int((dt<=0).sum()))
            md = read_sensor(r, 'Metadata.csv').iloc[0]
            row.update(手機=md['device name'], 平台=md['platform'])
            issues = []
            if not row['有陀螺儀']: issues.append('缺少 Gyroscope.csv')
            else:
                g = read_sensor(r, 'Gyroscope.csv', usecols=['seconds_elapsed','x','y','z'])
                gt = g['seconds_elapsed'].to_numpy(dtype=float)
                if len(gt)<2 or not np.isfinite(gt).all() or (np.diff(gt)<=0).any(): issues.append('陀螺儀時間戳異常')
                if not np.isfinite(g[['x','y','z']].to_numpy(dtype=float)).all(): issues.append('陀螺儀數值異常')
            if (dt<=0).any(): issues.append('加速度時間戳非遞增')
            if not np.isfinite(a[['x','y','z']].to_numpy(dtype=float)).all(): issues.append('加速度數值異常')
            row['狀態'] = '；'.join(issues) or 'OK'
        except Exception as e:
            row['狀態'] = f'讀取異常: {e}'
        rows.append(row)
    if not rows:
        print(f'找不到可辨識的資料：{DATA_DIR}'); return
    inv = pd.DataFrame(rows)
    path = OUT_DIR / '資料盤點.csv'
    inv.to_csv(path,index=False,encoding='utf-8-sig')
    print(inv.to_string(index=False))
    for col in ['手機','位置']:
        if col in inv: print(f'\n=== 類別 x {col}（錄製筆數）===\n',pd.crosstab(inv['類別'],inv[col]))
    if '分鐘' in inv: print('\n=== 各類別總分鐘 ===\n',inv.groupby('類別')['分鐘'].sum())
    print('\n輸出：',path)

if __name__ == '__main__': main()
