from pathlib import Path
import os
import zipfile
from upload_groups import group_uploads
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from data_loader import parse_upload
from inference import infer, predict_frames
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
        groups = group_uploads(files)
        truth = request.form.get("ground_truth", "")
        truth = truth if truth in ("mrt", "bus", "car") and len(groups) == 1 else None
        results = []
        for name, filename, raw in groups:
            try:
                result = parse_upload(filename, raw)
                result.update(name=name, ground_truth=truth, phone_position=request.form.get("phone_position", ""))
                try:
                    result["inference"] = infer(filename, raw, truth)
                except Exception:
                    app.logger.exception("Inference failed")
                    result["inference"] = {"status":"error", "message":"模型推論失敗，請查看終端機。"}
                results.append(result)
            except (ValueError, UnicodeDecodeError, OSError) as exc:
                results.append({"name":name,"error":str(exc)})
        return jsonify(trips=results, **(results[0] if len(results)==1 and "error" not in results[0] else {}))
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
        return jsonify(predict_frames(*frames,mode=mode))
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
