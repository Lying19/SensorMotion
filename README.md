# SensorMotion v0.5.1 — 統一模型環境版

本版延續 v0.5 的 Flask 網站、三模型檔案辨識、12 種 Sensor Logger 資料分析、Location 地圖與雙欄圖表。**尚未加入 Android 即時感測器讀取。**

## 1. Windows / VS Code 安裝與啟動

建議使用 **Python 3.11** 建立乾淨虛擬環境，避免舊環境殘留不同版本的 scikit-learn。PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import sklearn, numpy, pandas, scipy, joblib; print(sklearn.__version__, numpy.__version__, pandas.__version__, scipy.__version__, joblib.__version__)"
python backend/app.py
```

開啟 http://127.0.0.1:5000。若原本 Flask 已啟動，請先 Ctrl+C 停止。

**版本：** scikit-learn 1.8.0、NumPy 2.3.5、pandas 2.2.3、SciPy 1.17.0、joblib 1.5.3。Flask 3.1.3。

## 2. 模型訓練方式

三模型均使用 `tools/features.csv`（原始研究已產生的 23,698 筆視窗特徵），28 個 `acc_`/`gyr_` 特徵，並按 `trip` 做 Median/IQR 標準化。

- RandomForest：300 trees、`class_weight=balanced`、`random_state=0`，使用全部視窗。
- SVM：StandardScaler + RBF SVC（C=3、`class_weight=balanced`、`probability=True`），使用固定種子 0 抽樣的 8,000 筆視窗。
- kNN：StandardScaler + KNeighborsClassifier(15)，使用相同 8,000 筆視窗。

已產生 `backend/models/final_model.joblib`、`svm_model.joblib`、`knn_model.joblib` 及 `model_manifest.json`。可在同一環境中重新訓練：

```powershell
python tools/retrain_models.py --features tools/features.csv
```

**注意：** 此步驟是部署用模型訓練，並非重新執行 leave-one-recording-out 交叉驗證。原本研究報告的 Accuracy / Macro-F1 不能直接視為新模型的獨立測試成績。`probability=True` 會改變 SVM 的機率輸出方式；機率也不等於準確率。

## 3. 測試方式

在「檔案辨識」上傳單一行程的 Sensor Logger ZIP（需有 Accelerometer.csv 與 Gyroscope.csv），選擇真實交通工具，再到「辨識結果比較」查看三個模型。建議至少三分鐘完整行程。

已使用測試 ZIP 驗證 Python 推論流程可回傳三模型結果；**本環境沒有安裝 Flask，未完成瀏覽器端到端驗證**。請在 Windows 實測網站功能。

## 4. Render 部署提示

建議建置指令 `pip install -r requirements.txt`，啟動指令 `gunicorn backend.app:app --bind 0.0.0.0:$PORT`（請先確認 app.py 暴露 `app` 物件）。Render 環境請使用 Python 3.11；此專案含已訓練的 joblib 檔，部署前確認 Git 儲存庫檔案大小限制、模型載入記憶體及上傳資料隱私。**切勿從不可信來源載入 joblib。**

## 5. 目前限制

新蒐集資料的辨識表現仍待驗證；Android 即時辨識尚未實作。現有完整行程 Median/IQR 標準化不可直接照搬成每 2 秒即時預測。


## v0.6 Android 即時感測器讀取（實驗版）

- 「即時辨識」頁籤已改為手機感測器讀取與視覺化，不執行 RF/SVM/kNN 即時預測。
- Android Chrome 使用 `devicemotion` 讀取 acceleration（優先）及 rotationRate，顯示最近 20 秒的加速度和陀螺儀圖表、事件取樣率、筆數與最新數值。
- 可開始、停止、清除及下載 CSV。每次開始為新紀錄；單次上限 180,000 筆。
- **Android 手機必須使用 HTTPS**。電腦的 `http://127.0.0.1:5000` 僅適合在電腦本機開啟；手機透過 `http://電腦區網IP:5000` 不屬安全來源，通常無法使用感測器。可先用 HTTPS 測試隧道或部署至 HTTPS 站點。
- `rotationRate` 為度/秒，換算弧度/秒；對應 beta→X、gamma→Y、alpha→Z，實機方向和座標定義仍需校驗。
- 若 `acceleration` 不存在，改以 `accelerationIncludingGravity` 顯示，CSV `acc_source` 會標示 `with_gravity`；**不可直接視為模型訓練所用線性加速度**。
- 事件取樣率是最近 5 秒收到的 `devicemotion` 事件頻率，不保證感測器實際硬體取樣率，也不保證達到 50 Hz。
- 為避免背景錄製資料缺漏，切到背景會自動停止；CSV 僅儲存本機記憶體，重新整理即消失。
- 原有 v0.5.1 三模型 ZIP 辨識、感測器圖表及 GPS 地圖保留。
