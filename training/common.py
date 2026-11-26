"""共用研究設定與原始資料讀取（ZIP / 資料夾）。"""
import os
import re
import zipfile
from pathlib import Path, PurePosixPath
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get('DATA_DIR', ROOT / 'data'))
OUT_DIR = Path(os.environ.get('OUT_DIR', ROOT / 'outputs'))
FS = 50
WIN_SEC = float(os.environ.get('WIN_SEC', 2.0))
OVERLAP = 0.5
MAX_GAP = 0.5
TRIM_SEC = 30
WIN = int(WIN_SEC * FS)
STEP = int(WIN * (1 - OVERLAP))
CLASS_NAMES = {'MRT': '捷運', 'bus': '公車', 'car': '汽車'}
POS_NAMES = {'hand': '手持', 'pocket': '口袋', 'bag': '包包'}

def labels(name):
    low = name.lower()
    cls = next((c for c in CLASS_NAMES if c.lower() in low), None)
    pos = next((p for p in POS_NAMES if p in low), None)
    return cls, pos

def parse_date_tag(name):
    """
    解析錄製資料名稱中的日期，統一回傳 MMDD 格式。

    優先使用完整日期 YYYY-MM-DD，
    若沒有完整日期，再使用檔名開頭的 MMDD。
    """

    # 格式一：包含完整日期 YYYY-MM-DD
    # 例如：16_bus_bag-2026-11-15_22-50-33
    m = re.search(r'\d{4}-(\d{2})-(\d{2})', name)

    if m:
        return m.group(1) + m.group(2)

    # 格式二：檔名開頭為 MMDD
    # 例如：1114_car_pocket5
    m = re.match(r'(\d{4})(?=_)', name)

    if m:
        return m.group(1)

    # 無法辨識日期
    return ''

def find_recordings(root=DATA_DIR):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    records = {}
    def add(name, source, folder, archive=None):
        cls, pos = labels(name)
        if not cls or not pos:
            print(f'[略過] 無法解析標籤: {source}'); return
        key = name.casefold()
        if key in records:
            print(f'[重名略過，請確認是否同一趟] {source}'); return
        records[key] = dict(trip=name, cls=cls, position=pos,
                    date_tag=parse_date_tag(name), source=str(source),
                    folder=folder, archive=archive)
    for p in sorted(root.rglob('Accelerometer.csv')):
        add(p.parent.name, p.parent, p.parent)
    for z in sorted(root.rglob('*.zip')):
        try:
            with zipfile.ZipFile(z) as f:
                for member in sorted(f.namelist()):
                    p = PurePosixPath(member)
                    if p.name != 'Accelerometer.csv' or '__MACOSX' in p.parts: continue
                    folder = p.parent
                    name = folder.name
                    if not all(labels(name)): name = z.stem
                    add(name, f'{z}!{folder}', folder, z)
        except (OSError, zipfile.BadZipFile) as e:
            print(f'[ZIP 錯誤] {z}: {e}')
    return list(records.values())

def read_sensor(record, filename, **kwargs):
    path = record['folder'] / filename
    if record['archive'] is None:
        return pd.read_csv(path, **kwargs)
    with zipfile.ZipFile(record['archive']) as z:
        with z.open(str(path)) as f:
            return pd.read_csv(f, **kwargs)

def has_sensor(record, filename):
    path = record['folder'] / filename
    if record['archive'] is None: return path.is_file()
    with zipfile.ZipFile(record['archive']) as z:
        return str(path) in z.namelist()

def normalize_per_trip(X, cols):
    g = X.groupby('trip')[cols]
    med, q1, q3 = g.median(), g.quantile(.25), g.quantile(.75)
    Z = X.copy(); t = X['trip'].values
    Z[cols] = (X[cols].values - med.loc[t].values) / ((q3-q1).loc[t].values + 1e-9)
    return Z
