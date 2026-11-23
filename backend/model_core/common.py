"""共用設定：所有步驟都讀這個檔案。要調整參數只改這裡。"""
import os, re, glob
import numpy as np

# ---- 路徑（相對於你放腳本的資料夾）----
DATA_DIR = os.environ.get("DATA_DIR", "data")        # 把四個壓縮檔解開後的「所有錄製資料夾」放在這裡（可以有巢狀資料夾）
OUT_DIR = os.environ.get("OUT_DIR", "outputs")      # 結果輸出位置

# ---- 前處理參數 ----
FS = 50                  # 重新取樣頻率 (Hz)
WIN_SEC = float(os.environ.get("WIN_SEC", 2.0))            # 視窗長度 (秒)
OVERLAP = 0.5            # 視窗重疊比例
MAX_GAP = 0.5            # 原始資料中斷超過這個秒數的視窗直接丟掉
TRIM_SEC = 30            # 每趟錄製頭尾各裁掉幾秒（上下車、走路、等車）
WIN = int(WIN_SEC * FS)
STEP = int(WIN * (1 - OVERLAP))

CLASS_NAMES = {"MRT": "捷運", "bus": "公車", "car": "汽車"}
POS_NAMES = {"hand": "手持", "pocket": "口袋", "bag": "包包"}

def find_recordings(root=DATA_DIR):
    """找出所有含 Accelerometer.csv 的資料夾，並由資料夾名稱解析類別與手機位置。"""
    recs = []
    for acc in sorted(glob.glob(os.path.join(root, "**", "Accelerometer.csv"), recursive=True)):
        d = os.path.dirname(acc)
        name = os.path.basename(d)
        low = name.lower()
        cls = "bus" if "bus" in low else "car" if "car" in low else "mrt" if "mrt" in low else None
        if cls is None:
            print(f"[略過] 資料夾名稱看不出類別: {name}"); continue
        cls = "MRT" if cls == "mrt" else cls
        pos = "hand" if "hand" in low else "pocket" if "pocket" in low else "bag" if "bag" in low else None
        if pos is None:
            print(f"[略過] 資料夾名稱看不出手機位置: {name}"); continue
        m = re.match(r"(\d{4})", name)
        recs.append(dict(trip=name, dir=d, cls=cls, position=pos, date_tag=m.group(1) if m else ""))
    return recs

def normalize_per_trip(X, cols):
    """每趟錄製各自標準化：減該趟中位數、除以該趟 IQR（降低手機型號造成的整體幅度差異）。"""
    g = X.groupby("trip")[cols]
    med, q1, q3 = g.median(), g.quantile(.25), g.quantile(.75)
    Z = X.copy(); t = X["trip"].values
    Z[cols] = (X[cols].values - med.loc[t].values) / ((q3 - q1).loc[t].values + 1e-9)
    return Z
