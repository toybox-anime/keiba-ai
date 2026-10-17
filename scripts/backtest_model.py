"""勝率モデル vs 市場オッズ の検証（時系列で学習/テストを分割）.

指標:
  winner logloss : 勝ち馬に付けた確率の -log の平均（小さいほど良い）。確率の当て方そのもの
  的中率         : 各レースで一番確率が高い馬が勝った割合
  単勝回収率     : p×オッズ>閾値 の馬に100円ずつ賭けたときの回収率（最終オッズで払戻を近似）
"""
import sys, math, json, random
from collections import defaultdict
sys.path.insert(0, "src")
import numpy as np
import lightgbm as lgb
from keiba_ai.dataset import load_dataset
from keiba_ai.ml import FEATURE_NAMES

BANEI = "0304"  # 帯広ばんえい
rows = load_dataset("data/dataset.jsonl")
races = defaultdict(list)
for r in rows:
    races[r["race_id"]].append(r)

clean = {}
for rid, hs in races.items():
    odds = [h["features"].get("odds_win") for h in hs]
    if any(o is None or (isinstance(o, float) and math.isnan(o)) or o <= 0 for o in odds):
        continue
    if sum(h["is_win"] for h in hs) != 1 or len(hs) < 5:
        continue  # 同着・頭数不足は除外
    clean[rid] = sorted(hs, key=lambda h: h["num"])
print(f"レース総数 {len(races)} / 検証に使えるレース {len(clean)}")


def market_p(hs):
    inv = np.array([1 / h["features"]["odds_win"] for h in hs])
    return inv / inv.sum()


def mat(hs):
    return np.array([[h["features"][k] for k in FEATURE_NAMES] for h in hs], dtype=float)


def softmax(s, t):
    z = np.array(s) / t
    z = np.exp(z - z.max())
    return z / z.sum()


def evaluate(name, test_ids, probfn, out):
    ll, hit, bets = [], 0, defaultdict(lambda: [0, 0.0])
    for rid in test_ids:
        hs = clean[rid]
        p = probfn(rid, hs)
        w = next(i for i, h in enumerate(hs) if h["is_win"])
        ll.append(-math.log(max(p[w], 1e-9)))
        hit += int(np.argmax(p) == w)
        for th in (1.0, 1.1, 1.3):
            for i, h in enumerate(hs):
                if p[i] * h["features"]["odds_win"] > th:
                    bets[th][0] += 1
                    if i == w:
                        bets[th][1] += h["features"]["odds_win"]
    out[name] = {"ll": np.array(ll)}
    roi = " ".join(
        f"EV>{th}: {b[0]}点 回収{(b[1] / b[0] * 100 if b[0] else 0):.0f}%" for th, b in sorted(bets.items())
    )
    print(f"  {name:<22} logloss {np.mean(ll):.4f}  的中率 {hit / len(test_ids) * 100:.1f}%  | {roi}")


def run(split_date, label, exclude_banei):
    ids = sorted(rid for rid in clean if not (exclude_banei and rid[8:12] == BANEI))
    train_ids = [r for r in ids if r[:8] < split_date]
    test_ids = [r for r in ids if r[:8] >= split_date]
    # 学習の最後15%を温度・木の数の調整用に取り分ける（テストは一切見ない）
    k = int(len(train_ids) * 0.85)
    fit_ids, val_ids = train_ids[:k], train_ids[k:]
    print(f"\n=== {label}: 学習 {len(train_ids)}R（〜{split_date}前） / テスト {len(test_ids)}R"
          f"{'（ばんえい除外）' if exclude_banei else ''}")

    def stack(id_list):
        X = np.vstack([mat(clean[r]) for r in id_list])
        rel = np.concatenate([[2 if h["is_win"] else (1 if h["is_top3"] else 0) for h in clean[r]] for r in id_list])
        win = np.concatenate([[h["is_win"] for h in clean[r]] for r in id_list])
        return X, rel, win, [len(clean[r]) for r in id_list]

    out = {}
    evaluate("市場オッズ", test_ids, lambda rid, hs: market_p(hs), out)

    # A) 今のツールと同じ LambdaRank（温度は検証用で調整）
    Xf, relf, winf, gf = stack(fit_ids)
    rank = lgb.train({"objective": "lambdarank", "metric": "ndcg", "learning_rate": 0.05,
                      "num_leaves": 31, "min_data_in_leaf": 20, "verbose": -1},
                     lgb.Dataset(Xf, label=relf, group=gf), num_boost_round=200)

    def ll_on(ids_, fn):
        return np.mean([-math.log(max(fn(r, clean[r])[next(i for i, h in enumerate(clean[r]) if h["is_win"])], 1e-9)) for r in ids_])

    best_t = min((0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0),
                 key=lambda t: ll_on(val_ids, lambda r, hs: softmax(rank.predict(mat(hs)), t)))
    evaluate(f"A ランキング(T={best_t})", test_ids, lambda rid, hs: softmax(rank.predict(mat(hs)), best_t), out)

    # B) 勝ち/負けの二値分類（レース内で正規化）
    clf = lgb.train({"objective": "binary", "learning_rate": 0.03, "num_leaves": 15,
                     "min_data_in_leaf": 50, "feature_fraction": 0.8, "verbose": -1},
                    lgb.Dataset(Xf, label=winf), num_boost_round=300)

    def clf_p(rid, hs):
        q = clf.predict(mat(hs))
        return q / q.sum()
    evaluate("B 二値分類", test_ids, clf_p, out)

    # C) 市場確率を土台に、モデルで少しだけ補正（p ∝ 市場^a × モデル^b、a,bは検証用で調整）
    grid = [(a, b) for a in (0.8, 0.9, 1.0, 1.1) for b in (0.0, 0.1, 0.2, 0.3, 0.5)]

    def blend(a, b):
        def f(rid, hs):
            m = market_p(hs)
            q = clf.predict(mat(hs)); q = q / q.sum()
            z = a * np.log(m) + b * np.log(np.maximum(q, 1e-9))
            z = np.exp(z - z.max())
            return z / z.sum()
        return f
    a, b = min(grid, key=lambda ab: ll_on(val_ids, blend(*ab)))
    evaluate(f"C 市場+補正(a={a},b={b})", test_ids, blend(a, b), out)

    # 市場との差の信頼区間（レース単位のブートストラップ）
    base = out["市場オッズ"]["ll"]
    rng = np.random.default_rng(0)
    for name, v in out.items():
        if name == "市場オッズ":
            continue
        d = v["ll"] - base
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        verdict = "市場より良い" if hi < 0 else ("市場より悪い" if lo > 0 else "差なし（誤差の範囲）")
        print(f"    {name:<22} 市場との差 {d.mean():+.4f} [95%区間 {lo:+.4f}〜{hi:+.4f}] → {verdict}")


run("20261001", "直近テスト(10月)", exclude_banei=True)
run("20260915", "長めのテスト(9/15〜)", exclude_banei=True)
run("20260915", "ばんえい込み(9/15〜)", exclude_banei=False)
