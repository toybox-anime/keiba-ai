"""「オッズにない情報」（近走の中身・血統など）で市場オッズを補正できるかの検証（Issue #6）.

近走5走の中身から指標を作り、時系列で学習/テストに分けて、市場オッズ（ev.fair_win_probs と同じ補正）
の勝率予測より勝ち馬をうまく当てられるか（winner logloss）を比べる。

  B: 市場の確率(log) ＋ 新しい指標 の二値分類（レース内で正規化）
  C: 新しい指標だけのモデル q を作り、p ∝ 市場^a × q^b で混ぜる（a,b は学習の後半で調整）

使い方（リポジトリ直下で）:  python scripts/backtest_details.py [テスト開始日 YYYYMMDD]
"""
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, "src")
from keiba_ai import details  # noqa: E402
from keiba_ai.dataset import load_dataset  # noqa: E402
from keiba_ai.ev import WIN_ALPHA  # noqa: E402

BANEI = "0304"
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "20260915"
NAN = float("nan")

rows = load_dataset("data/dataset.jsonl")
D = details.load_all(Path("."))
races = defaultdict(list)
for r in rows:
    races[r["race_id"]].append(r)


def day(s):
    return datetime.strptime(s, "%Y%m%d")


# ---- タイムの基準（場×距離×馬場ごとの平均と標準偏差）を学習期間の近走だけで作る ----
def key3(p):
    return (p["track"], p["distance"], p["going"])


def key2(p):
    return (p["track"], p["distance"])


acc3, acc2, accl = defaultdict(list), defaultdict(list), defaultdict(list)
for rid, horses in D.items():
    if rid[:8] >= SPLIT:
        continue
    for h in horses.values():
        for p in h["past"]:
            if p["time"] and p["distance"] and p["date"] and p["date"] < SPLIT:
                acc3[key3(p)].append(p["time"])
                acc2[key2(p)].append(p["time"])
                if p["last3f"]:
                    accl[key2(p)].append(p["last3f"])


def stat(v):
    a = np.array(v)
    return (float(a.mean()), float(a.std() or 1.0)) if len(a) >= 30 else None


st3 = {k: s for k, v in acc3.items() if (s := stat(v))}
st2 = {k: s for k, v in acc2.items() if (s := stat(v))}
stl = {k: s for k, v in accl.items() if (s := stat(v))}


def speed(p):
    """スピード指数：同じ場・距離（できれば馬場も）の平均より何σ速いか（大きいほど速い）."""
    if not p["time"]:
        return NAN
    s = st3.get(key3(p)) or st2.get(key2(p))
    return (s[0] - p["time"]) / s[1] if s else NAN


def finish_last3f(p):
    s = stl.get(key2(p))
    return (s[0] - p["last3f"]) / s[1] if (s and p["last3f"]) else NAN


def mean(v):
    v = [x for x in v if not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else NAN


def best(v):
    v = [x for x in v if not (isinstance(x, float) and math.isnan(x))]
    return float(max(v)) if v else NAN


NEW_FEATURES = [
    "sf_last", "sf_mean", "sf_best", "sf_trend", "margin_last", "margin_mean", "margin_best",
    "l3f_last", "l3f_mean", "early_pos", "late_gain", "days_off", "dist_change", "n_past",
    "beat_pop_last", "beat_pop_mean", "pop_last", "course_win", "course_top3", "course_n",
    "dist_win", "dist_top3", "dist_n", "age_days",
]


def horse_new_features(rid, num, cur_dist):
    h = D.get(rid, {}).get(num)
    f = dict.fromkeys(NEW_FEATURES, NAN)
    if not h:
        return f
    past = [p for p in h["past"] if p["date"] and p["date"] < rid[:8]]
    sfs = [speed(p) for p in past]
    signed = [(-p["margin"] if p["finish"] == 1 else p["margin"]) for p in past if p["margin"] is not None]
    l3 = [finish_last3f(p) for p in past]
    f.update(
        sf_last=sfs[0] if sfs else NAN, sf_mean=mean(sfs), sf_best=best(sfs),
        sf_trend=(sfs[0] - mean(sfs[1:])) if len(sfs) >= 2 else NAN,
        margin_last=signed[0] if signed else NAN, margin_mean=mean(signed),
        margin_best=min(signed) if signed else NAN,
        l3f_last=l3[0] if l3 else NAN, l3f_mean=mean(l3), n_past=len(past),
    )
    pas = [p for p in past if p["passing"] and p["n_runners"]]
    if pas:
        f["early_pos"] = mean([p["passing"][0] / p["n_runners"] for p in pas])
        f["late_gain"] = mean([(p["passing"][0] - (p["finish"] or p["passing"][-1])) / p["n_runners"] for p in pas])
    if past:
        f["days_off"] = (day(rid[:8]) - day(past[0]["date"])).days
        if past[0]["distance"] and cur_dist:
            f["dist_change"] = cur_dist - past[0]["distance"]
        bp = [(p["popularity"] - p["finish"]) for p in past if p["popularity"] and p["finish"]]
        f["beat_pop_last"] = bp[0] if bp else NAN
        f["beat_pop_mean"] = mean(bp)
        f["pop_last"] = past[0]["popularity"] or NAN
    for name, rec in (("course", h["course_rec"]), ("dist", h["dist_rec"])):
        if rec:
            w, s2, t3, o = rec[0]
            n = w + s2 + t3 + o
            f[f"{name}_n"] = n
            if n:
                f[f"{name}_win"] = w / n
                f[f"{name}_top3"] = (w + s2 + t3) / n
    if h.get("birth"):
        f["age_days"] = (day(rid[:8]) - day(h["birth"])).days
    return f


# ---- レースごとの行列を作る ----
OLD = ["impost", "body_weight", "body_change", "rate1", "rate2", "distance_m", "going_code", "n_runners", "num"]
data = []
for rid, hs in races.items():
    if rid[8:12] == BANEI or len(hs) < 5 or rid not in D:
        continue
    odds = [h["features"].get("odds_win") for h in hs]
    if any(o is None or (isinstance(o, float) and math.isnan(o)) or o <= 0 for o in odds):
        continue
    if sum(h["is_win"] for h in hs) != 1:
        continue
    hs = sorted(hs, key=lambda h: h["num"])
    w = np.array([(1 / h["features"]["odds_win"]) ** WIN_ALPHA for h in hs])
    m = w / w.sum()
    cur_dist = hs[0]["features"].get("distance_m")
    feats = []
    for h, mi in zip(hs, m):
        nf = horse_new_features(rid, h["num"], cur_dist)
        feats.append([math.log(mi)] + [h["features"].get(k, NAN) for k in OLD] + [nf[k] for k in NEW_FEATURES])
    data.append({"rid": rid, "m": m, "X": np.array(feats, dtype=float), "y": np.array([h["is_win"] for h in hs])})

COLS = ["log_market"] + OLD + NEW_FEATURES
train = [d for d in data if d["rid"][:8] < SPLIT]
test = [d for d in data if d["rid"][:8] >= SPLIT]
k = int(len(train) * 0.85)
fit, val = train[:k], train[k:]
print(f"学習 {len(train)}R / テスト {len(test)}R（{SPLIT}〜、ばんえい除外）  指標 {len(COLS)}個")


def stack(ds, cols):
    return np.vstack([d["X"][:, cols] for d in ds]), np.concatenate([d["y"] for d in ds])


def ll(ds, probfn):
    return np.array([-math.log(max(probfn(d)[d["y"].argmax()], 1e-12)) for d in ds])


def norm(q):
    return q / q.sum()


params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 100,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1}


def train_model(ds, cols, rounds=600):
    X, y = stack(ds, cols)
    return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=rounds)


all_cols = list(range(len(COLS)))
no_odds = [i for i, c in enumerate(COLS) if c != "log_market"]

# B: 市場(log) + 新指標。木の数は検証用で選ぶ
best_r, best_v = None, 1e9
for r in (150, 300, 600, 1000):
    mdl = train_model(fit, all_cols, r)
    v = ll(val, lambda d: norm(mdl.predict(d["X"][:, all_cols]))).mean()
    if v < best_v:
        best_r, best_v = r, v
mB = train_model(train, all_cols, best_r)

# C: 新指標だけのモデル q を作り、市場と混ぜる
mq_fit = train_model(fit, no_odds, 600)
grid = [(a, b) for a in (0.8, 0.9, 1.0, 1.1) for b in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6)]


def blend(mq, a, b):
    def f(d):
        q = norm(mq.predict(d["X"][:, no_odds]))
        z = a * np.log(d["m"]) + b * np.log(np.maximum(q, 1e-9))
        z = np.exp(z - z.max())
        return z / z.sum()
    return f


a, b = min(grid, key=lambda ab: ll(val, blend(mq_fit, *ab)).mean())
mq = train_model(train, no_odds, 600)

results = {
    "市場オッズ": lambda d: d["m"],
    f"B 市場+新指標(木{best_r})": lambda d: norm(mB.predict(d["X"][:, all_cols])),
    f"C 混合(a={a},b={b})": blend(mq, a, b),
    "参考: 新指標だけ": lambda d: norm(mq.predict(d["X"][:, no_odds])),
}
base = ll(test, results["市場オッズ"])
rng = np.random.default_rng(0)
for name, fn in results.items():
    v = ll(test, fn)
    hit = np.mean([fn(d).argmax() == d["y"].argmax() for d in test]) * 100
    line = f"  {name:<24} logloss {v.mean():.4f}  的中率 {hit:.1f}%"
    if name != "市場オッズ":
        diff = v - base
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        verdict = "市場より良い" if hi < 0 else ("市場より悪い" if lo > 0 else "差なし（誤差の範囲）")
        line += f"  市場との差 {diff.mean():+.4f} [95%区間 {lo:+.4f}〜{hi:+.4f}] → {verdict}"
    print(line)

imp = mB.feature_importance("gain")
order = np.argsort(-imp)[:12]
print("\nB で効いている指標（上位12）:", ", ".join(f"{COLS[i]}({imp[i] / imp.sum() * 100:.0f}%)" for i in order))
