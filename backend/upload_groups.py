"""Group recordings without extracting untrusted archives to disk."""
import io
import zipfile
from pathlib import PurePosixPath
from data_loader import SENSORS, MAX_ZIP_TOTAL

def pack(items):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, raw in items.items():
            z.writestr(name, raw)
    return out.getvalue()

def group_uploads(files):
    groups = []
    csvs = {}
    for file in files:
        name = PurePosixPath(file.filename.replace('\\', '/')).name
        raw = file.read()
        ext = PurePosixPath(name).suffix.lower()
        if ext == '.zip':
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                if sum(i.file_size for i in z.infolist()) > MAX_ZIP_TOTAL:
                    raise ValueError('ZIP 解壓縮後資料過大')
                folders = {}
                for i in z.infolist():
                    p = PurePosixPath(i.filename.replace('\\', '/'))
                    if p.is_absolute() or '..' in p.parts:
                        raise ValueError('ZIP 包含不安全路徑')
                    if i.is_dir() or p.suffix.lower() != '.csv' or p.stem not in SENSORS:
                        continue
                    if i.file_size > 50*1024*1024:
                        raise ValueError('單一感測器檔案超過 50 MB')
                    items = folders.setdefault(str(p.parent), {})
                    if p.name in items:
                        raise ValueError('ZIP 內有重複感測器檔案')
                    items[p.name] = z.read(i)
                for folder, items in folders.items():
                    groups.append((name + (' / '+folder if folder!='.' else ''), 'recording.zip', items))
        elif ext == '.csv':
            if name in csvs:
                raise ValueError('CSV 檔名重複；不同行程請分別壓成 ZIP')
            if len(raw)>50*1024*1024:
                raise ValueError('單一 CSV 超過 50 MB')
            csvs[name] = raw
        elif ext == '.json':
            groups.append((name,name,raw))
        else:
            raise ValueError('不支援的檔案格式')
    if csvs:
        groups.append(('CSV 行程資料','recording.zip',csvs))
    if not groups:
        raise ValueError('未找到感測器行程資料')
    return groups
