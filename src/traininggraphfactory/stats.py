"""稳健统计、去趋势与步类型分层。

**检出模块与资源图共用这一套基线**，这不是复用癖：两处各写一遍，迟早会算出两个
不同的中位数，于是「这一步是异常」和「这一步的瓶颈资源」互相矛盾，而报告里
没有任何东西能解释这个矛盾。

五条约定，每一条都对应一种会把检出搞废的方式：

1. **稳健而非均值**。一次 loss spike 会把均值和标准差一起拽走，于是异常自己
   把自己的基线抬高了 —— 用 `median` / `MAD` 才不会。
2. **必须去趋势**。训练 loss 从 4.2 衰减到 0.12（34 倍）。拿全量的中位数当基线，
   loss 在前段永远"偏高"、后段永远"偏低"，真正的 spike 被埋在趋势里检不出来。
   这里用 **Theil–Sen**（两两斜率的中位数）拟合趋势，它对 <29% 的离群点免疫。
3. **强趋势信号走乘性通道**。跨度超过 `detrend_span` 倍的信号在对数空间里做 ——
   因为「这一步是平时的 3 倍」和「这一步比平时多 2」是两种不同的异常观，
   而乘性噪声在对数空间才是同方差的。
4. **MAD 会退化，必须设地板**。ckpt 步的 `disk_mbps` 九步里有八步完全相同，
   MAD = 0，z 会直接爆成无穷。地板按 `mad_floor_ratio × |变换后中位数|` 取。
5. **必须按步类型分层**。ckpt / eval 步的 `step_time_s` 与 `disk_mbps` 天生比
   训练步高 3~30 倍。不分层的话，**每一个 ckpt 步都会被标成异常**（全是假阳性）。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from .schema import StepSignal

__all__ = [
    "median",
    "mad",
    "robust_scale",
    "theil_sen",
    "step_kind",
    "strata",
    "Baseline",
    "fit_baseline",
    "stratified_baselines",
]

# 信号跨度超过这么多倍就走对数（乘性）通道
DETREND_SPAN = 4.0


def median(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError("median of empty")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def mad(xs: Sequence[float], center: float) -> float:
    """中位绝对偏差。乘 1.4826 后才是正态下的 σ 一致估计。"""
    if not xs:
        return 0.0
    return median([abs(x - center) for x in xs])


def robust_scale(residuals: Sequence[float], center_t: float, floor_ratio: float) -> float:
    """`max(1.4826·MAD(残差), floor_ratio·|变换后中位数|)`。

    第二项就是那个「地板」：MAD 退化到 0 时它兜底，否则 z 会爆成无穷。
    注意残差的中位数按构造接近 0，所以地板只能拿**变换后中位数**来定尺度，
    不能拿残差自己 —— 那会得到 0。
    """
    raw = 1.4826 * mad(residuals, 0.0)
    floor = abs(floor_ratio * center_t)
    scale = max(raw, floor)
    if scale <= 0.0:
        scale = 1e-12
    return scale


def theil_sen(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """`y = a + b·x` 的 Theil–Sen 估计：两两斜率取中位数，截距取残差中位数。

    选它而不是最小二乘，是因为最小二乘会被**它要检测的那些异常**拽偏 ——
    一次 18 倍的梯度爆炸足以让最小二乘的斜率翻个方向。
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return 0.0, ys[0]
    # 两两斜率直接建成列表。这是 O(n²) 的循环，越简单越好：列表推导省掉
    # 每轮的名字查找与 append，斜率顺序与逐对 (i, j) 的遍历顺序完全一致，
    # 因此后面的中位数逐位不变。
    slopes = [
        (ys[j] - ys[i]) / dx
        for i in range(n - 1)
        for j in range(i + 1, n)
        if (dx := xs[j] - xs[i]) != 0.0
    ]
    b = median(slopes) if slopes else 0.0
    a = median([y - b * x for x, y in zip(xs, ys)])
    return b, a


def step_kind(ss: StepSignal) -> str:
    """这一步属于哪一类。判据来自**事件本身**，不是配置里的周期数字 ——
    周期可能被改，而「这一步真的写了 checkpoint」是事实。

    **同时写 checkpoint 又跑评测的步单独成层。** 这是本项目最容易被忽略的一类步：
    它比纯 ckpt 步多出一整段评测时间，所以在任一个单层里都是离群点。
    本项目里 `ckpt_every=12` 与 `eval_every=10` 每 60 步撞一次，于是第 60 步
    既写 ckpt 又跑 eval —— 它的耗时比普通 ckpt 步长 30%，在 ckpt 层里 z≈15。
    """
    ckpt = ss.get("ckpt_write_s") is not None
    ev = ss.get("eval_loss") is not None
    if ckpt and ev:
        return "ckpt+eval"
    if ckpt:
        return "ckpt"
    if ev:
        return "eval"
    return "train"


def strata(step_table: Iterable[StepSignal]) -> dict[str, list[StepSignal]]:
    out: dict[str, list[StepSignal]] = {}
    for ss in step_table:
        out.setdefault(step_kind(ss), []).append(ss)
    return out


class Baseline:
    """一个信号在**一个步类型**上的稳健基线：趋势 + 尺度。

    变换空间里 `t = transform(x)`，残差 `r = t − (a + b·step)`，`z = r / scale`。
    `log_space=True` 时 `transform = ln`，于是 z 是**倍数**的度量：
    z=1 大致对应「是平时的 e^scale 倍」。
    """

    __slots__ = ("signal", "kind", "n", "log_space", "slope", "intercept", "scale", "center", "center_t")

    def __init__(
        self,
        signal: str,
        kind: str,
        n: int,
        log_space: bool,
        slope: float,
        intercept: float,
        scale: float,
        center: float,
        center_t: float,
    ) -> None:
        self.signal = signal
        self.kind = kind
        self.n = n
        self.log_space = log_space
        self.slope = slope
        self.intercept = intercept
        self.scale = scale
        self.center = center
        self.center_t = center_t

    @property
    def transform(self) -> str:
        return "log" if self.log_space else "raw"

    def expected(self, step: int) -> float:
        """这一步**本来应该**是多少（信号自己的量纲）。"""
        t = self.intercept + self.slope * step
        return math.exp(t) if self.log_space else t

    def z(self, x: float, step: int) -> float:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            # NaN 不走 z。**靠容差比较判 NaN 永远是 False**，所以这里必须原样返回
            # NaN，让调用方显式处理它。
            return float("nan")
        if self.log_space:
            if x <= 0.0:
                return float("nan")
            t = math.log(x)
        else:
            t = x
        return (t - (self.intercept + self.slope * step)) / self.scale

    def to_dict(self) -> dict[str, object]:
        return {
            "signal": self.signal,
            "kind": self.kind,
            "n": self.n,
            "transform": self.transform,
            "slope": self.slope,
            "intercept": self.intercept,
            "scale": self.scale,
            "center": self.center,
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"Baseline({self.signal}@{self.kind}, {self.transform}, "
            f"center={self.center:.4g}, scale={self.scale:.4g}, slope={self.slope:.3g}, n={self.n})"
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Baseline":
        """从产物重建。`center_t` 不在产物里 —— 它只是拟合时的中间量，
        重算一遍比多存一个字段更不容易漂。"""
        slope = float(raw["slope"])
        intercept = float(raw["intercept"])
        center = float(raw["center"])
        center_t = math.log(center) if str(raw["transform"]) == "log" else center
        return cls(
            signal=str(raw["signal"]),
            kind=str(raw["kind"]),
            n=int(raw["n"]),
            log_space=str(raw["transform"]) == "log",
            slope=slope,
            intercept=intercept,
            scale=float(raw["scale"]),
            center=center,
            center_t=center_t,
        )


def fit_baseline(
    signal: str,
    kind: str,
    samples: Sequence[tuple[int, float]],
    floor_ratio: float,
    *,
    detrend: bool = True,
    span: float = DETREND_SPAN,
) -> Baseline | None:
    """`samples` 是 `(步, 值)`。**只拿有限值拟合** ——
    让 NaN 参与基线，等于让异常定义基线。
    """
    pairs = [(s, v) for s, v in samples if v is not None and not math.isnan(v)]
    if not pairs:
        return None
    xs = [float(s) for s, _v in pairs]
    ys = [float(v) for _s, v in pairs]
    center = median(ys)

    log_space = False
    if detrend and len(pairs) >= 3 and min(ys) > 0.0:
        log_space = (max(ys) / min(ys)) >= span

    ts = [math.log(y) for y in ys] if log_space else ys
    if detrend:
        slope, intercept = theil_sen(xs, ts)
    else:
        slope, intercept = 0.0, median(ts)
    residuals = [t - (intercept + slope * x) for t, x in zip(ts, xs)]
    center_t = median(ts)
    scale = robust_scale(residuals, center_t, floor_ratio)

    return Baseline(signal, kind, len(pairs), log_space, slope, intercept, scale, center, center_t)


def stratified_baselines(
    step_table: Sequence[StepSignal],
    signals: Sequence[str],
    floor_ratio: float,
    *,
    min_stratum_size: int = 1,
    fallback_kind: str = "all",
    detrend: bool = True,
    span: float = DETREND_SPAN,
) -> dict[tuple[str, str], Baseline]:
    """`(信号, 步类型) → 基线`。

    样本太少的层（本项目里 ckpt 层只有 9 步）退到 `fallback_kind` 的全量基线上 ——
    拿 3 个点去估 MAD 得到的不是一个尺度，是一个巧合。
    """
    groups = strata(step_table)
    out: dict[tuple[str, str], Baseline] = {}
    for sig in signals:
        overall = [(ss.step, ss.get(sig)) for ss in step_table]
        b_all = fit_baseline(sig, fallback_kind, overall, floor_ratio, detrend=detrend, span=span)
        if b_all is not None:
            out[(sig, fallback_kind)] = b_all
        for kind, members in groups.items():
            if len(members) < min_stratum_size:
                continue
            b = fit_baseline(
                sig, kind, [(ss.step, ss.get(sig)) for ss in members], floor_ratio, detrend=detrend, span=span
            )
            if b is not None:
                out[(sig, kind)] = b
    return out
