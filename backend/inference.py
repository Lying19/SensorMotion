"""Full-trip inference with RF, SVM, kNN; uses teammate preprocessing."""
import io
import sys
import zipfile
from functools import lru_cache
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

@lru_cache(maxsize=1)
def load_model(name="RandomForest"):
    path = MODEL_DIR / MODEL_FILES[name]
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {name} 模型檔：{path.name}")
    # Only load trusted, locally bundled models; never deserialize user uploads.
    return joblib.load(path)

def _read_csv_pair(filename, content):
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
    data, error = _read_csv_pair(filename, content)
    if error:
        return {"status": "unavailable", "message": error}
    a = data["Accelerometer.csv"]
    g = data["Gyroscope.csv"]
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
    bundles = {name: load_model(name) for name in MODEL_FILES}
    cols = bundles["RandomForest"]["cols"]
    windows = pd.DataFrame(rows)
    if not set(cols).issubset(windows.columns):
        return {"status": "unavailable", "message": "模型特徵欄位不相容。"}
    windows["trip"] = "uploaded_trip"
    z = common.normalize_per_trip(windows, cols)
    x = z[cols].replace([np.inf, -np.inf], np.nan)
    if x.isna().any().any():
        return {"status": "unavailable", "message": "特徵存在無效數值，無法預測。"}
    outputs = {}
    for name, bundle in bundles.items():
        model_cols = list(bundle["cols"])
        if model_cols != list(cols):
            raise ValueError(f"{name} 特徵順序與 RandomForest 不一致")
        model = bundle["model"]
        # SVC(probability=True) and kNN both expose predict_proba.
        mean = model.predict_proba(x[model_cols]).mean(axis=0)
        classes = list(model.classes_)
        label = classes[int(np.argmax(mean))]
        outputs[name] = {
            "model": name, "predicted": label,
            "predicted_zh": LABELS.get(label, label),
            "correct": (label.lower() == ground_truth.lower()) if ground_truth else None,
            "probabilities": {c: round(float(prob), 5) for c, prob in zip(classes, mean)},
            "window_count": len(rows)
        }
    rf = outputs["RandomForest"]
    return {"status": "ok", **rf, "models": outputs,
            "note": "機率為整趟有效視窗平均模型輸出，不是準確率；三模型採每趟標準化。RF 使用全部 23698 個訓練視窗；SVM/kNN 各使用 8000 個視窗。三模型統一以 scikit-learn 1.8.0 重新訓練，並非留一趟驗證結果。"}
