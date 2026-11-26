"""步驟 3：訓練與評估三類辨識 baseline（捷運 / 公車 / 汽車）。
重點：以「整趟錄製」為單位做 leave-one-recording-out —— 同一趟的視窗不會同時出現在訓練與測試。
實驗 A：全部資料。  實驗 B：只用 iPhone（避免模型靠「手機型號」作弊）。"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from common import *

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC", "Noto Sans CJK JP", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
LABELS = ["MRT", "bus", "car"]; ZH = [CLASS_NAMES[c] for c in LABELS]
MAX_TRAIN = 8000                       # SVM/kNN 訓練視窗上限（僅抽樣訓練集以加速，可能影響結果）

def models():
    return {"RandomForest": RandomForestClassifier(300, class_weight="balanced", n_jobs=-1, random_state=0),
            "SVM(RBF)": make_pipeline(StandardScaler(), SVC(C=3, class_weight="balanced")),
            "kNN(k=15)": make_pipeline(StandardScaler(), KNeighborsClassifier(15))}

def loro(X, y, groups, make):
    rng = np.random.default_rng(0); pred = np.empty(len(y), dtype=object)
    for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(X, y, groups), 1):
        assert not set(groups[tr]) & set(groups[te])
        if fold == 1 or fold % 10 == 0 or fold == len(np.unique(groups)):
            print(f"  LORO {fold}/{len(np.unique(groups))}", flush=True)
        if make is not None and len(tr) > MAX_TRAIN and not isinstance(make, RandomForestClassifier):
            tr = rng.choice(tr, MAX_TRAIN, replace=False)
        from sklearn.base import clone
        m = clone(make).fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return pred

def evaluate(D, cols, tag):
    X, y, g = D[cols].values, D["cls"].values, D["trip"].values
    res, preds = [], {}
    for name, m in models().items():
        print(f"開始 {tag} / {name}", flush=True)
        p = loro(X, y, g, m); preds[name] = p
        rec = recall_score(y, p, labels=LABELS, average=None, zero_division=0)
        # 趟級別：每趟取視窗預測的多數決
        T = pd.DataFrame(dict(trip=g, y=y, p=p)); tv = T.groupby("trip").agg(y=("y", "first"), p=("p", lambda s: s.value_counts().index[0]))
        res.append(dict(實驗=tag, 模型=name, 視窗準確率=round(accuracy_score(y, p), 3), 視窗MacroF1=round(f1_score(y, p, labels=LABELS, average="macro"), 3),
                        捷運召回=round(rec[0], 3), 公車召回=round(rec[1], 3), 汽車召回=round(rec[2], 3),
                        趟級準確率=round((tv.y == tv.p).mean(), 3), 趟數=len(tv)))
        print(res[-1], flush=True)
        pd.DataFrame(res).to_csv(OUT_DIR / ("結果暫存_" + tag[0] + ".csv"), index=False, encoding="utf-8-sig")
    return pd.DataFrame(res), preds

def plot_cm(y, p, title, path):
    cm = confusion_matrix(y, p, labels=LABELS); cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4.4)); ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i,j]*100:.0f}%\n({cm[i,j]})", ha="center", va="center", color="white" if cmn[i, j] > .5 else "black", fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(ZH); ax.set_yticks(range(3)); ax.set_yticklabels(ZH)
    ax.set_xlabel("預測"); ax.set_ylabel("實際"); ax.set_title(title, fontsize=10); plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

if __name__ == "__main__":
    X = pd.read_csv(os.path.join(OUT_DIR, "features.csv")); cols = [c for c in X.columns if c.startswith(("acc_", "gyr_"))]
    if len(cols) != 28 or not np.isfinite(X[cols].to_numpy()).all():
        raise ValueError("預期 28 個有限數值特徵")
    if X[["trip", "cls", "position", "device"]].isna().any().any() or X.groupby("trip").cls.nunique().max() != 1:
        raise ValueError("行程標籤異常")
    print("手機 x 類別（視窗數）："); print(pd.crosstab(X["cls"], X["device"]))
    # ---- 實驗 A：全部資料 ----
    RA, PA = evaluate(X, cols, "A 全部資料")
    # ---- 實驗 B：只用 iPhone ----
    XI = X[X["device"].str.contains("iPhone")].reset_index(drop=True)
    RB, PB = evaluate(XI, cols, "B 只用iPhone")
    R = pd.concat([RA, RB]); R.to_csv(os.path.join(OUT_DIR, "結果總表.csv"), index=False, encoding="utf-8-sig")
    # ---- 混淆矩陣（以 RandomForest）----
    plot_cm(X["cls"].values, PA["RandomForest"], "實驗A（全部資料）RandomForest\nleave-one-recording-out", os.path.join(OUT_DIR, "混淆矩陣_A.png"))
    plot_cm(XI["cls"].values, PB["RandomForest"], "實驗B（只用iPhone）RandomForest\nleave-one-recording-out", os.path.join(OUT_DIR, "混淆矩陣_B.png"))
    # ---- 手機位置影響：各位置的辨識率（實驗 A, RandomForest）----
    T = X[["trip", "cls", "position", "device"]].copy(); T["pred"] = PA["RandomForest"]; T["ok"] = T["cls"] == T["pred"]
    pos = T.groupby(["cls", "position"])["ok"].agg(["mean", "size"]).round(3).rename(columns={"mean": "視窗準確率", "size": "視窗數"}); pos.to_csv(os.path.join(OUT_DIR, "位置影響_A.csv"), encoding="utf-8-sig"); print(pos)
    per = T.groupby("trip").agg(類別=("cls", "first"), 位置=("position", "first"), 手機=("device", "first"), 準確率=("ok", "mean"), 視窗數=("ok", "size")).round(3)
    per.to_csv(os.path.join(OUT_DIR, "每趟結果_A.csv"), encoding="utf-8-sig")
    # ---- 特徵重要性（RandomForest 全資料）----
    rf = RandomForestClassifier(300, class_weight="balanced", n_jobs=-1, random_state=0).fit(X[cols], X["cls"])
    imp = pd.Series(rf.feature_importances_, cols).sort_values(ascending=False); imp.head(10).round(3).to_csv(os.path.join(OUT_DIR, "特徵重要性前10.csv"), header=["重要性"], encoding="utf-8-sig"); print(imp.head(10).round(3))
