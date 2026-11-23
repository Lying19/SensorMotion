from pathlib import Path
import os
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from data_loader import parse_upload
from inference import infer

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")

@app.post("/api/upload")
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="請先選擇檔案。"), 400
    try:
        raw = file.read()
        result = parse_upload(file.filename, raw)
        result["ground_truth"] = request.form.get("ground_truth", "") if request.form.get("ground_truth", "") in ("mrt", "bus", "car") else None
        result["phone_position"] = request.form.get("phone_position", "")
        try:
            result["inference"] = infer(file.filename, raw, result["ground_truth"])
        except Exception as exc:
            app.logger.exception("Inference failed")
            result["inference"] = {"status": "error", "message": "模型推論發生錯誤，請查看 Flask 終端機紀錄。"}
        return jsonify(result)
    except (ValueError, UnicodeDecodeError, OSError) as exc:
        return jsonify(error=str(exc)), 400

@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify(error="檔案超過 100 MB 限制。"), 413

if __name__ == "__main__":
    app.run(debug=False, host=os.environ.get("SENSORMOTION_HOST", "127.0.0.1"), port=5000)
