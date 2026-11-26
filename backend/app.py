from pathlib import Path
import os
import io
from time import perf_counter
import zipfile
from upload_groups import group_uploads
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from data_loader import read_upload, summarize_upload, SENSORS
from inference import predict_frames
import pandas as pd
from predict import FIELDS

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")

@app.post("/api/upload")
def upload():
    files = request.files.getlist("file")
    if not files or any(not f.filename for f in files):
        return jsonify(error="請先選擇檔案。"), 400
    try:
        started = perf_counter()
        groups = group_uploads(files)
        app.logger.warning("upload grouping %.3fs", perf_counter()-started)
        truth = request.form.get("ground_truth", "")
        truth = truth if truth in ("mrt", "bus", "car") and len(groups) == 1 else None
        results = []
        for name, filename, raw in groups:
            try:
                t = perf_counter()
                frames = {Path(k).stem: pd.read_csv(io.BytesIO(v), low_memory=False) if v else pd.DataFrame() for k,v in raw.items() if Path(k).stem in SENSORS} if isinstance(raw,dict) else read_upload(filename,raw)
                parse_sec = perf_counter()-t
                t = perf_counter()
                result = summarize_upload(frames)
                preview_sec = perf_counter()-t
                result.update(name=name, ground_truth=truth, phone_position=request.form.get("phone_position", ""))
                try:
                    t = perf_counter()
                    if isinstance(raw,dict) and all(k in frames for k in ("Accelerometer","Gyroscope")):
                        result["inference"] = predict_frames(frames["Accelerometer"][FIELDS],frames["Gyroscope"][FIELDS],truth=truth)
                    else:
                        result["inference"] = {"status":"unavailable","message":"辨識需要同一趟 Accelerometer.csv 與 Gyroscope.csv。"}
                    app.logger.warning("upload parse=%.3fs preview=%.3fs inference=%.3fs",parse_sec,preview_sec,perf_counter()-t)
                except Exception:
                    app.logger.exception("Inference failed")
                    result["inference"] = {"status":"error", "message":"模型推論失敗，請查看終端機。"}
                results.append(result)
            except (ValueError, UnicodeDecodeError, OSError) as exc:
                results.append({"name":name,"error":str(exc)})
        t = perf_counter()
        response = jsonify(trips=results, **(results[0] if len(results)==1 and "error" not in results[0] else {}))
        app.logger.warning("upload serialization=%.3fs total=%.3fs",perf_counter()-t,perf_counter()-started)
        return response
    except (ValueError, UnicodeDecodeError, OSError, zipfile.BadZipFile) as exc:
        return jsonify(error=str(exc)), 400

@app.post('/api/live/predict')
def live_predict():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict): raise ValueError('需要 JSON 物件')
        mode = data.get('mode', 'live')
        if mode not in ('live', 'trip'): raise ValueError('mode 不正確')
        frames = []
        for key in ('acc','gyro'):
            rows = data.get(key)
            if not isinstance(rows,list) or len(rows)>180000: raise ValueError('感測器資料格式或筆數不正確')
            if any(not isinstance(r,list) or len(r)!=4 for r in rows): raise ValueError('每筆需要時間與三軸數值')
            frames.append(pd.DataFrame(rows,columns=FIELDS))
        return jsonify(predict_frames(*frames,mode=mode,session=data.get("session"),model=data.get("model", "all")))
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify(error=str(exc)),400
    except Exception:
        app.logger.exception('Live inference failed')
        return jsonify(error='模型推論失敗，請查看伺服器紀錄'),500

@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify(error="檔案超過 100 MB 限制。"), 413

if __name__ == "__main__":
    app.run(debug=False, host=os.environ.get("SENSORMOTION_HOST", "127.0.0.1"), port=5000)
