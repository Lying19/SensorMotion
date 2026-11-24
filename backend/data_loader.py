"""Sensor Logger ZIP / CSV / JSON preview parser (no model inference)."""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = ["Activity", "Gravity", "TotalAcceleration", "Battery", "Accelerometer", "AccelerometerUncalibrated", "Gyroscope", "GyroscopeUncalibrated", "Orientation", "Location", "Annotation", "Metadata"]
MAX_POINTS = 2500
MAX_ZIP_TOTAL = 240 * 1024 * 1024


def _time(df):
    if "seconds_elapsed" in df.columns:
        return pd.to_numeric(df["seconds_elapsed"], errors="coerce")
    if "time" in df.columns:
        t = pd.to_numeric(df["time"], errors="coerce")
        valid = t.dropna()
        if len(valid) and valid.abs().median() > 1e15:
            return (t - valid.iloc[0]) / 1e9
        return t - valid.iloc[0] if len(valid) else t
    return pd.Series(np.arange(len(df), dtype=float), index=df.index)


def _summarize(df, sensor):
    if sensor == "Metadata":
        # Avoid exposing device identifiers and other unnecessary private metadata.
        allowed = {"version", "device name", "recording time", "recording timezone", "platform", "appVersion", "sampleRateMs", "standardisation", "platform version"}
        row = df.iloc[0].to_dict() if len(df) else {}
        return {"sensor": sensor, "kind": "metadata", "count": len(df), "duration_seconds": None, "estimated_hz": None,
                "metadata": {str(k): str(v)[:150] for k, v in row.items() if k in allowed and pd.notna(v)}}
    if df.empty:
        return {"sensor": sensor, "kind": "empty", "count": 0, "duration_seconds": 0, "estimated_hz": None, "time": [], "series": []}
    df = df.copy()
    df["_elapsed"] = _time(df)
    df = df.dropna(subset=["_elapsed"]).sort_values("_elapsed").drop_duplicates("_elapsed")
    if df.empty:
        raise ValueError("時間欄位沒有有效資料")
    times = df["_elapsed"].to_numpy(dtype=float)
    times = times - times[0]
    dt = np.diff(times)
    dt = dt[dt > 0]
    hz = round(float(1 / np.median(dt)), 2) if len(dt) else None
    indices = np.linspace(0, len(df) - 1, min(len(df), MAX_POINTS), dtype=int)
    small = df.iloc[indices]
    sampled_time = np.round(times[indices], 3).tolist()
    base = {"sensor": sensor, "count": len(df), "duration_seconds": round(float(times[-1]), 2),
            "estimated_hz": hz, "plotted_points": len(indices), "time": sampled_time}
    if sensor == "Location":
        lat = pd.to_numeric(df.get("latitude"), errors="coerce") if "latitude" in df else pd.Series(dtype=float)
        lon = pd.to_numeric(df.get("longitude"), errors="coerce") if "longitude" in df else pd.Series(dtype=float)
        if len(lat) != len(df) or len(lon) != len(df):
            raise ValueError("Location 缺少 latitude 或 longitude")
        valid = lat.between(-90, 90) & lon.between(-180, 180) & ~((lat == 0) & (lon == 0))
        loc = df.loc[valid].copy()
        if len(loc) > MAX_POINTS:
            loc = loc.iloc[np.linspace(0, len(loc) - 1, MAX_POINTS, dtype=int)]
        points = []
        for _, r in loc.iterrows():
            p = {"lat": round(float(r["latitude"]), 7), "lon": round(float(r["longitude"]), 7), "t": round(float(r["_elapsed"] - df["_elapsed"].iloc[0]), 2)}
            for field in ("speed", "horizontalAccuracy", "altitude", "bearing"):
                if field in r and pd.notna(r[field]):
                    p[field] = round(float(r[field]), 3)
            points.append(p)
        speed = pd.to_numeric(small["speed"], errors="coerce") if "speed" in small else pd.Series(dtype=float)
        return {**base, "kind": "location", "points": points, "valid_location_count": int(valid.sum()),
                "series": [{"name": "speed", "values": [round(float(v), 3) if pd.notna(v) else None for v in speed]}] if len(speed) else []}
    if sensor in ("Activity", "Annotation"):
        field = "activity" if sensor == "Activity" else next((c for c in df.columns if c not in ("time", "seconds_elapsed", "_elapsed")), None)
        events = [{"t": round(float(r["_elapsed"] - df["_elapsed"].iloc[0]), 2), "value": str(r[field])[:160]}
                  for _, r in small.iterrows() if field and pd.notna(r[field])]
        return {**base, "kind": "events", "events": events, "field": field}
    if sensor == "Battery":
        names = [c for c in ("batteryLevel", "voltage", "chargingCurrent") if c in df]
        states = [{"t": round(float(r["_elapsed"] - df["_elapsed"].iloc[0]), 2), "value": str(r["batteryState"])}
                  for _, r in small.iterrows() if "batteryState" in df and pd.notna(r["batteryState"])]
    elif sensor == "Orientation":
        names = [c for c in ("roll", "pitch", "yaw", "qx", "qy", "qz", "qw") if c in df]
        states = []
    else:
        names = [c for c in ("x", "y", "z") if c in df]
        states = []
    if not names:
        raise ValueError("找不到可繪圖的數值欄位")
    series = []
    for c in names:
        vals = pd.to_numeric(small[c], errors="coerce")
        series.append({"name": c, "values": [round(float(v), 5) if pd.notna(v) else None for v in vals]})
    if all(c in df for c in ("x", "y", "z")):
        xyz = small[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        magnitude = np.linalg.norm(xyz, axis=1)
        series.append({"name": "Magnitude", "values": [round(float(v), 5) if np.isfinite(v) else None for v in magnitude]})
    return {**base, "kind": "numeric", "series": series, "states": states}


def parse_upload(filename, content):
    ext = Path(filename).suffix.lower()
    datasets = {}
    results, errors = {}, {}
    quality = None
    def summarize(sensor, df):
        try:
            results[sensor] = _summarize(df, sensor)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            errors[sensor] = str(exc)
    if ext == ".zip":
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                members = archive.infolist()
                if sum(x.file_size for x in members) > MAX_ZIP_TOTAL:
                    raise ValueError("ZIP 解壓縮後資料過大（超過 240 MB）")
                selected = {}
                for item in members:
                    sensor = Path(item.filename).stem
                    if item.is_dir() or Path(item.filename).suffix.lower() != ".csv" or sensor not in SENSORS:
                        continue
                    if item.file_size > 50 * 1024 * 1024:
                        continue
                    selected[sensor] = item
                for sensor in SENSORS:
                    if sensor not in selected: continue
                    item = selected[sensor]
                    with archive.open(item) as f:
                        summarize(sensor, pd.read_csv(f, low_memory=False) if item.file_size else pd.DataFrame())
        except (zipfile.BadZipFile, RuntimeError) as exc:
            raise ValueError("ZIP 檔案無法讀取") from exc
    elif ext == ".csv":
        sensor = Path(filename).stem
        from browser_csv import is_browser_csv, read_browser
        if is_browser_csv(content):
            datasets, quality = read_browser(content)
        elif sensor in SENSORS:
            datasets[sensor] = pd.read_csv(io.BytesIO(content), low_memory=False)
        else:
            raise ValueError("請上傳 SensorMotion CSV 或以感測器名稱命名的 CSV")
    elif ext == ".json":
        try:
            obj = json.loads(content)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("JSON 格式無法解析") from exc
        if not isinstance(obj, list):
            raise ValueError("JSON 應為 Sensor Logger 資料陣列")
        grouped = {s: [] for s in SENSORS}
        for row in obj:
            if isinstance(row, dict) and row.get("sensor") in grouped:
                grouped[row["sensor"]].append(row)
        datasets = {s: pd.DataFrame(rows) for s, rows in grouped.items() if rows}
    else:
        raise ValueError("只支援 ZIP、CSV、JSON")
    for sensor in SENSORS:
        if sensor not in datasets:
            continue
        try:
            results[sensor] = _summarize(datasets[sensor], sensor)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            errors[sensor] = str(exc)
    if not results:
        raise ValueError("未找到可用的 Sensor Logger 資料。" + str(errors))
    return {"sensors": list(results), "datasets": results, "skipped": errors, "supported": SENSORS, "quality": quality}
