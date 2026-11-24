"""Full-trip inference with RF, SVM, kNN; uses teammate preprocessing."""
import io
import sys
import zipfile
import gc
import logging
from threading import Lock
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CORE = Path(__file__).resolve().parent / "model_core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
import common
from importlib import import_module
preprocess = import_module("02_preprocess_features")

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_FILES = {"RandomForest": "final_model.joblib", "SVM": "svm_model.joblib", "kNN": "knn_model.joblib"}
LABELS = {"MRT": "捷運", "bus": "公車", "car": "汽車"}

LOGGER = logging.getLogger(__name__)
INFERENCE_LOCK = Lock()
BATCH_SIZE = 128

def log_memory(stage):
    # Linux / Render: current process RSS, not total instance memory.
    try:
        with open('/proc/self/status') as f:
            rss = next(line for line in f if line.startswith('VmRSS:'))
        LOGGER.warning('memory stage=%s rss_mib=%.1f', stage, int(rss.split()[1])/1024)
    except (OSError, StopIteration, ValueError):
        pass

def load_model(name="RandomForest"):
    path = MODEL_DIR / MODEL_FILES[name]
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {name} 模型檔：{path.name}")
    # Only load trusted, locally bundled models; never deserialize user uploads.
    return joblib.load(path)

def _read_csv_pair(filename, content):
    if filename.lower().endswith('.csv'):
        from browser_csv import is_browser_csv
        if is_browser_csv(content):
            return None, "瀏覽器 CSV 目前僅提供感測器分析；跨平台模型尚未驗證。"
    if not filename.lower().endswith(".zip"):
        return None, "模型辨識需要同一趟的 Accelerometer.csv 和 Gyroscope.csv，請上傳 ZIP；單檔仍可做感測器分析。"
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            items = {}
            for item in z.infolist():
                name = Path(item.filename).name
                if name in ("Accelerometer.csv", "Gyroscope.csv") and not item.is_dir():
                    if item.file_size > 50 * 1024 * 1024:
                        return None, f"{name} 超過 50 MB 上限"
                    if name in items:
                        return None, f"ZIP 內有多個 {name}，請只上傳單趟錄製"
                    items[name] = item
            if len(items) != 2:
                return None, "ZIP 缺少 Accelerometer.csv 或 Gyroscope.csv，僅提供感測器分析。"
            data = {}
            for name, item in items.items():
                with z.open(item) as stream:
                    df = pd.read_csv(stream, usecols=["seconds_elapsed", "x", "y", "z"])
                for col in ("seconds_elapsed", "x", "y", "z"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.replace([np.inf, -np.inf], np.nan).dropna().sort_values("seconds_elapsed").drop_duplicates("seconds_elapsed")
                if len(df) < 2:
                    return None, f"{name} 沒有足夠有效資料"
                data[name] = df
            return data, None
    except (zipfile.BadZipFile, ValueError, KeyError, pd.errors.ParserError) as exc:
        return None, f"模型資料格式不符：{exc}"

def infer(filename, content, ground_truth):
    with INFERENCE_LOCK:
        return _infer(filename, content, ground_truth)

def _infer(filename, content, ground_truth):
    log_memory('inference_start')
    data, error = _read_csv_pair(filename, content)
    if error:
        return {"status": "unavailable", "message": error}
    a = data["Accelerometer.csv"]
    g = data["Gyroscope.csv"]
    if any(float(d.seconds_elapsed.iloc[-1]-d.seconds_elapsed.iloc[0]) > 10800 for d in (a,g)):
        return {"status":"unavailable","message":"單趟模型分析上限為 3 小時。"}
    ma, va = preprocess.mag_resampled(a)
    mg, vg = preprocess.mag_resampled(g)
    if ma is None or mg is None:
        return {"status": "unavailable", "message": "行程過短：前後各裁 30 秒後不足 2 秒視窗。"}
    n = min(len(ma), len(mg))
    valid = va[:n] & vg[:n]
    rows = []
    for start in range(0, n - common.WIN + 1, common.STEP):
        if not valid[start:start + common.WIN].all():
            continue
        row = {}
        row.update(preprocess.feats(ma[start:start + common.WIN], "acc"))
        row.update(preprocess.feats(mg[start:start + common.WIN], "gyr"))
        rows.append(row)
    if len(rows) < 20:
        return {"status": "unavailable", "message": f"有效視窗只有 {len(rows)} 個（至少需要 20 個）。"}
    del data, a, g, ma, mg, va, vg, valid
    gc.collect()
    cols = None
    windows = pd.DataFrame(rows)
    windows["trip"] = "uploaded_trip"
    window_count = len(rows)
    del rows
    outputs = {}
    for name in MODEL_FILES:
        log_memory(name + '_before_load')
        bundle = load_model(name)
        model = bundle['model']
        try:
            log_memory(name + '_loaded')
            model_cols = list(bundle['cols'])
            if cols is None:
                cols = model_cols
                if not set(cols).issubset(windows.columns):
                    return {"status": "unavailable", "message": "模型特徵欄位不相容。"}
                z = common.normalize_per_trip(windows, cols)
                x = z[cols].replace([np.inf, -np.inf], np.nan)
                del z, windows
                if x.isna().any().any():
                    return {"status": "unavailable", "message": "特徵存在無效數值，無法預測。"}
            if model_cols != cols:
                raise ValueError(f"{name} 特徵順序與 RandomForest 不一致")
            # Avoid RF parallel prediction and large temporary probability arrays.
            params = model.get_params(deep=True)
            jobs = {k: 1 for k in params if k == 'n_jobs' or k.endswith('__n_jobs')}
            if jobs:
                model.set_params(**jobs)
            total = np.zeros(len(model.classes_), dtype=np.float64)
            for start in range(0, len(x), BATCH_SIZE):
                total += model.predict_proba(x.iloc[start:start+BATCH_SIZE]).sum(axis=0)
            mean = total / len(x)
            classes = list(model.classes_)
            label = classes[int(np.argmax(mean))]
            outputs[name] = {
                "model": name, "predicted": label,
                "predicted_zh": LABELS.get(label, label),
                "correct": (label.lower() == ground_truth.lower()) if ground_truth else None,
                "probabilities": {c: round(float(prob), 5) for c, prob in zip(classes, mean)},
                "window_count": window_count
            }
        finally:
            del model, bundle
            gc.collect()
            log_memory(name + '_released')
    rf = outputs["RandomForest"]
    return {"status": "ok", **rf, "models": outputs,
            "note": "機率為整趟有效視窗平均模型輸出，不是準確率；三模型採每趟標準化。RF 使用全部 23698 個訓練視窗；SVM/kNN 各使用 8000 個視窗。三模型統一以 scikit-learn 1.8.0 重新訓練，並非留一趟驗證結果。"}
