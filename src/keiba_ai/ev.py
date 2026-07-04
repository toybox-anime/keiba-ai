"""期待値(EV)とケリー基準の計算エンジン（純粋な数学・ネット不要）.

単勝オッズから「控除率を除いたフェア勝率」を作り、Harville モデルで
連系・三連系の的中確率を推定する。各買い目の EV とケリー比率を出す。

Harville モデル（独立性の近似）:
    P(i 1着)            = p_i
    P(i 1着, j 2着)     = p_i · p_j/(1−p_i)
    P(i,j,k この順)     = p_i · p_j/(1−p_i) · p_k/(1−p_i−p_j)
そこから:
    馬連(i,j)   = 上位2着が{i,j}（順不同）
    三連複(i,j,k)= 上位3着が{i,j,k}（順不同）
    ワイド(i,j) = i,j がともに3着以内
"""

from __future__ import annotations

from itertools import permutations


def fair_win_probs(odds_by_num: dict[int, float]) -> dict[int, float]:
    """単勝オッズ {馬番: オッズ} → 控除率を除いたフェア勝率 {馬番: p}（合計1）."""
    inv = {n: 1.0 / o for n, o in odds_by_num.items() if o and o > 0}
    total = sum(inv.values())
    if total <= 0:
        return {}
    return {n: v / total for n, v in inv.items()}


def exacta_prob(p: dict[int, float], i: int, j: int) -> float:
    """P(i 1着, j 2着)."""
    if i == j or i not in p or j not in p:
        return 0.0
    denom = 1.0 - p[i]
    return p[i] * (p[j] / denom) if denom > 1e-9 else 0.0


def trifecta_prob(p: dict[int, float], i: int, j: int, k: int) -> float:
    """P(i,j,k がこの順)."""
    if len({i, j, k}) < 3 or any(x not in p for x in (i, j, k)):
        return 0.0
    d1 = 1.0 - p[i]
    d2 = 1.0 - p[i] - p[j]
    if d1 <= 1e-9 or d2 <= 1e-9:
        return 0.0
    return p[i] * (p[j] / d1) * (p[k] / d2)


def quinella_prob(p: dict[int, float], i: int, j: int) -> float:
    """馬連: 上位2着が {i,j}（順不同）."""
    return exacta_prob(p, i, j) + exacta_prob(p, j, i)


def trio_prob(p: dict[int, float], i: int, j: int, k: int) -> float:
    """三連複: 上位3着が {i,j,k}（順不同）= 6通りの順列の和."""
    return sum(trifecta_prob(p, a, b, c) for a, b, c in permutations((i, j, k)))


def wide_prob(p: dict[int, float], i: int, j: int) -> float:
    """ワイド: i,j がともに3着以内 = Σ_k 三連複(i,j,k)."""
    others = [n for n in p if n not in (i, j)]
    return sum(trio_prob(p, i, j, k) for k in others)


def place_prob(p: dict[int, float], i: int) -> float:
    """複勝: i が3着以内に入る確率（1着＋2着＋3着）."""
    if i not in p:
        return 0.0
    others = [n for n in p if n != i]
    prob = p[i]                                              # 1着
    prob += sum(exacta_prob(p, a, i) for a in others)        # 2着
    prob += sum(trifecta_prob(p, a, b, i) for a in others for b in others if a != b)  # 3着
    return prob


def ev(odds: float, prob: float) -> float:
    """期待値（払戻倍率ベース）。EV>1 なら理論上プラス."""
    return round(odds * prob, 3)


def kelly_fraction(odds: float, prob: float) -> float:
    """ケリー比率 f* = p − (1−p)/b （b=odds−1）。負なら0（賭けない）."""
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = prob - (1.0 - prob) / b
    return max(0.0, round(f, 4))
