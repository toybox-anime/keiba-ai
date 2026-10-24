"""着順確率モデルの補正パラメータを推定・検証する.

ハーヴィル（1着確率から2・3着を比例配分）は人気馬の2・3着確率を過大評価することが
知られている。そこで次の補正（Henery/Stern 型の簡易版）を入れる:

  1着: p_i ∝ (単勝フェア確率)^α        … 本命・大穴の偏り（favorite-longshot bias）を補正
  2着: q2_j ∝ p_j^λ2（1着馬を除いて正規化）
  3着: q3_k ∝ p_k^λ3（1・2着馬を除いて正規化）

α, λ2, λ3 を学習期間の最尤推定で決め、テスト期間で「予測した当たる確率」と
「実際に当たった割合」が合うかを確認する。

使い方（リポジトリ直下で）:  python scripts/fit_place_model.py [テスト開始日 YYYYMMDD]
"""
import math
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

sys.path.insert(0, "src")
from keiba_ai.dataset import load_dataset  # noqa: E402

BANEI = "0304"
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "20260915"


def load_races():
    races = defaultdict(list)
    for r in load_dataset("data/dataset.jsonl"):
        races[r["race_id"]].append(r)
    out = []
    for rid, hs in races.items():
        if rid[8:12] == BANEI or len(hs) < 5:
            continue
        odds = np.array([h["features"].get("odds_win") or np.nan for h in hs], dtype=float)
        fin = [h.get("finish") for h in hs]
        if np.isnan(odds).any() or (odds <= 0).any():
            continue
        try:
            order = [fin.index(k) for k in (1, 2, 3)]  # 1〜3着の行番号（同着や欠損は除外）
        except ValueError:
            continue
        if any(fin.count(k) != 1 for k in (1, 2, 3)):
            continue
        inv = 1 / odds
        out.append({"rid": rid, "base": inv / inv.sum(), "order": order, "n": len(hs)})
    return out


def probs(base, a):
    p = base ** a
    return p / p.sum()


def loglik(races, a, l2, l3):
    ll = 0.0
    for r in races:
        p = probs(r["base"], a)
        w1, w2, w3 = r["order"]
        q2 = p ** l2
        q3 = p ** l3
        ll += math.log(p[w1])
        ll += math.log(q2[w2] / (q2.sum() - q2[w1]))
        ll += math.log(q3[w3] / (q3.sum() - q3[w1] - q3[w2]))
    return ll


def fit(train):
    # α は1着だけで決まる → 先に1次元で探す
    best_a = max(np.arange(0.80, 1.41, 0.02),
                 key=lambda a: sum(math.log(probs(r["base"], a)[r["order"][0]]) for r in train))
    grid = np.arange(0.40, 1.21, 0.05)
    best = max(((l2, l3) for l2 in grid for l3 in grid), key=lambda x: loglik(train, best_a, *x))
    return round(float(best_a), 2), round(float(best[0]), 2), round(float(best[1]), 2)


def top3_probs(p, l2, l3):
    """各馬の3着内確率と、2頭が両方3着内（ワイド）の確率行列."""
    n = len(p)
    q2, q3 = p ** l2, p ** l3
    place = np.zeros(n)
    wide = np.zeros((n, n))
    for i in range(n):
        s2 = q2.sum() - q2[i]
        for j in range(n):
            if j == i:
                continue
            pij = p[i] * q2[j] / s2
            s3 = q3.sum() - q3[i] - q3[j]
            for k in range(n):
                if k in (i, j):
                    continue
                pr = pij * q3[k] / s3
                place[i] += pr
                place[j] += pr
                place[k] += pr
                wide[i, j] += pr
                wide[i, k] += pr
                wide[j, k] += pr
    wide = wide + wide.T
    return place, wide


def calibration(test, a, l2, l3, label):
    pb = defaultdict(lambda: [0, 0.0, 0])
    wb = defaultdict(lambda: [0, 0.0, 0])
    for r in test:
        p = probs(r["base"], a)
        place, wide = top3_probs(p, l2, l3)
        top = set(r["order"])
        for i in range(r["n"]):
            b = min(int(place[i] * 10), 9)
            s = pb[b]; s[0] += 1; s[1] += place[i]; s[2] += i in top
        fav = list(np.argsort(-p)[:5])
        for i, j in combinations(fav, 2):
            q = wide[i, j]
            b = min(int(q * 10), 9)
            s = wb[b]; s[0] += 1; s[1] += q; s[2] += (i in top and j in top)
    print(f"\n--- {label}（α={a}, λ2={l2}, λ3={l3}）")
    for name, tb in (("複勝(3着内)", pb), ("ワイド(上位5頭の組)", wb)):
        err = sum(abs(s[1] - s[2]) for s in tb.values()) / sum(s[0] for s in tb.values()) * 100
        print(f"  {name}  平均ずれ {err:.1f}ポイント")
        for b in sorted(tb):
            s = tb[b]
            if b >= 3:
                print(f"    予測{b * 10}-{b * 10 + 10}%: {s[0]:>5}件 予測{s[1] / s[0] * 100:5.1f}% → 実際{s[2] / s[0] * 100:5.1f}%")


races = load_races()
train = [r for r in races if r["rid"][:8] < SPLIT]
test = [r for r in races if r["rid"][:8] >= SPLIT]
print(f"学習 {len(train)}R / テスト {len(test)}R（{SPLIT}〜、ばんえい除外）")
a, l2, l3 = fit(train)
print(f"推定: α={a}, λ2={l2}, λ3={l3}")
print(f"テスト対数尤度  ハーヴィル {loglik(test, 1, 1, 1):.1f} → 補正後 {loglik(test, a, l2, l3):.1f}（大きいほど良い）")
calibration(test, 1.0, 1.0, 1.0, "今のツール（ハーヴィル）")
calibration(test, a, l2, l3, "補正後")
