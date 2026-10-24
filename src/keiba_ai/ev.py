"""期待値(EV)とケリー基準の計算エンジン（純粋な数学・ネット不要）.

単勝オッズから「控除率を除いたフェア勝率」を作り、着順モデルで
連系・三連系の的中確率を推定する。各買い目の EV とケリー比率を出す。

着順モデル（ハーヴィルを補正した Henery/Stern 型の簡易版）:
    1着: p_i ∝ (単勝フェア確率)^WIN_ALPHA          … 本命・大穴の偏りを補正
    2着: q2_j = p_j^LAMBDA2 を1着馬を除いて正規化
    3着: q3_k = p_k^LAMBDA3 を1・2着馬を除いて正規化
    P(i,j,k この順) = p_i · q2_j/Σ_{≠i} q2 · q3_k/Σ_{≠i,j} q3
素のハーヴィル（λ=1）は人気馬の2・3着確率を過大評価する（ワイド予測60-70%→実際52%）。
係数は scripts/fit_place_model.py で推定（2026-10、学習2,803R・テスト1,598R、ばんえい除外）。
補正後はテスト期間でワイドの予測と実際のずれが平均3.5→0.7ポイントに縮小。
そこから:
    馬連(i,j)   = 上位2着が{i,j}（順不同）
    三連複(i,j,k)= 上位3着が{i,j,k}（順不同）
    ワイド(i,j) = i,j がともに3着以内
"""

from __future__ import annotations

from itertools import permutations

WIN_ALPHA = 1.04
LAMBDA2 = 0.75
LAMBDA3 = 0.60


def fair_win_probs(odds_by_num: dict[int, float]) -> dict[int, float]:
    """単勝オッズ {馬番: オッズ} → 控除率と本命・大穴の偏りを補正した勝率 {馬番: p}（合計1）."""
    w = {n: (1.0 / o) ** WIN_ALPHA for n, o in odds_by_num.items() if o and o > 0}
    total = sum(w.values())
    if total <= 0:
        return {}
    return {n: v / total for n, v in w.items()}


class _Order:
    """1つのレースの着順確率を速く計算するための前計算（2・3着の重みとその合計）."""

    def __init__(self, p: dict[int, float]) -> None:
        self.p = p
        self.q2 = {n: v ** LAMBDA2 for n, v in p.items()}
        self.q3 = {n: v ** LAMBDA3 for n, v in p.items()}
        self.s2 = sum(self.q2.values())
        self.s3 = sum(self.q3.values())

    def exacta(self, i: int, j: int) -> float:
        d = self.s2 - self.q2[i]
        return self.p[i] * self.q2[j] / d if d > 1e-12 else 0.0

    def trifecta(self, i: int, j: int, k: int) -> float:
        d = self.s3 - self.q3[i] - self.q3[j]
        return self.exacta(i, j) * self.q3[k] / d if d > 1e-12 else 0.0


def _valid(p: dict[int, float], *nums: int) -> bool:
    return len(set(nums)) == len(nums) and all(x in p for x in nums)


def exacta_prob(p: dict[int, float], i: int, j: int) -> float:
    """P(i 1着, j 2着)."""
    return _Order(p).exacta(i, j) if _valid(p, i, j) else 0.0


def trifecta_prob(p: dict[int, float], i: int, j: int, k: int) -> float:
    """P(i,j,k がこの順)."""
    return _Order(p).trifecta(i, j, k) if _valid(p, i, j, k) else 0.0


def quinella_prob(p: dict[int, float], i: int, j: int) -> float:
    """馬連: 上位2着が {i,j}（順不同）."""
    if not _valid(p, i, j):
        return 0.0
    o = _Order(p)
    return o.exacta(i, j) + o.exacta(j, i)


def trio_prob(p: dict[int, float], i: int, j: int, k: int) -> float:
    """三連複: 上位3着が {i,j,k}（順不同）= 6通りの順列の和."""
    if not _valid(p, i, j, k):
        return 0.0
    o = _Order(p)
    return sum(o.trifecta(a, b, c) for a, b, c in permutations((i, j, k)))


def wide_prob(p: dict[int, float], i: int, j: int) -> float:
    """ワイド: i,j がともに3着以内 = Σ_k 三連複(i,j,k)."""
    if not _valid(p, i, j):
        return 0.0
    o = _Order(p)
    return sum(
        o.trifecta(a, b, c)
        for k in p if k not in (i, j)
        for a, b, c in permutations((i, j, k))
    )


def box_hit_prob(p: dict[int, float], nums: tuple[int, ...] | list[int]) -> float:
    """ワイドボックス: nums のうち2頭以上が3着以内に入る（＝どれか1点は的中）確率."""
    box = set(nums)
    o = _Order(p)
    return sum(
        o.trifecta(a, b, c)
        for a, b, c in permutations(p, 3)
        if len(box & {a, b, c}) >= 2
    )


def place_prob(p: dict[int, float], i: int) -> float:
    """複勝: i が3着以内に入る確率（1着＋2着＋3着）."""
    if i not in p:
        return 0.0
    o = _Order(p)
    others = [n for n in p if n != i]
    prob = p[i]                                              # 1着
    prob += sum(o.exacta(a, i) for a in others)              # 2着
    prob += sum(o.trifecta(a, b, i) for a in others for b in others if a != b)  # 3着
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
