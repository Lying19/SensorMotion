"""Shared verified 15-second inference for uploads and live snapshots."""
import io
import zipfile
from pathlib import Path
from threading import Lock
import pandas as pd
from predict import Predictor, FIELDS

MODEL_DIR = Path(__file__).resolve().parent / 'models'
LOCK = Lock()
_engine = None

def predict_frames(acc, gyro, mode='trip', truth=None):
    global _engine
    with LOCK:
        if _engine is None:
            _engine = Predictor(MODEL_DIR)
            for model in _engine.models.values():
                jobs = {k:1 for k in model.get_params(deep=True) if k=='n_jobs' or k.endswith('__n_jobs')}
                if jobs: model.set_params(**jobs)
        result = _engine.predict(acc, gyro, mode=mode, model='all')
    q = result['quality']
    result['status'] = 'ok' if q['status']=='OK' else 'unavailable'
    result['message'] = {'insufficient_data':'資料不足以形成15秒視窗','no_valid_windows':'沒有有效視窗'}.get(q['status'])
    for m in result['models'].values():
        m['correct'] = (m['prediction'].lower()==truth.lower()) if truth and m['prediction'] and not m['tie'] else None
    return result

def infer(filename, content, ground_truth=None):
    if not filename.lower().endswith('.zip'):
        return {'status':'unavailable','message':'辨識需要同一趟 Accelerometer.csv 與 Gyroscope.csv。'}
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        frames=[]
        for name in ('Accelerometer.csv','Gyroscope.csv'):
            matches=[i for i in z.infolist() if Path(i.filename).name==name and not i.is_dir()]
            if len(matches)!=1: return {'status':'unavailable','message':f'缺少或重複 {name}'}
            if matches[0].file_size>50*1024*1024: raise ValueError('感測器檔案過大')
            frames.append(pd.read_csv(z.open(matches[0]),usecols=FIELDS))
    return predict_frames(*frames, truth=ground_truth)
