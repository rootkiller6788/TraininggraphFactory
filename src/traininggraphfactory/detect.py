"""异常检出。**稳健 z + 分层基线 + episode 聚合 + NaN 显式判定。**

四条不能省的规则，每一条都对应一种会把检出搞废的方式：

1. **稳健 z**，不是均值/标准差。一次 loss spike 会把均值和 σ 一起拽走 ——
   异常自己把自己的基线抬高了，于是它检测不到自己。
2. **按步类型分层**。ckpt / eval 步的 `step_time_s` 与 `disk_mbps` 天生比训练步
   高 3~30 倍。不分层的话**每一个 ckpt 步都会被标成异常**，全是假阳性，
   而真异常淹没在里面。这是本项目最容易踩的坑。
3. **episode 聚合**。连续被标记的步聚成一段。逐步报出来的话，一次 6 步的 lr spike
   会变成 6 条异常 —— 召回率立刻虚高，而「有几起异常」这个数就没意义了。
4. **NaN 显式判**。`abs(nan - median) > k * mad` **永远是 False** —— 拿 NaN 去比
   大小不会报错，只会静默漏掉整次训练里最严重的那类失败。所以 NaN 走单独的分支，
   不进 z 通道。

检出只报「哪一步的哪个信号不对」，**不报因果** —— 那是 `recover.py` 的事。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .schema import (
    SIGNAL_HIGH_IS_BAD,
    Anomaly,
    StepSignal,
    WHAT_OF_SIGNAL,
)
from .stats import Baseline, step_kind, stratified_baselines

__all__ = ["detect", "DetectResult", "z_table", "anomalies_of_signal"]

_FALLBACK_KIND = "all"


class DetectResult:
    """检出产物。`z` 与 `baselines` 都留着 —— 报告要能解释**为什么**某一步被标出来。"""

    __slots__ = ("anomalies", "z", "baselines", "threshold", "monitored", "skipped")

    def __init__(
        self,
        anomalies: tuple[Anomaly, ...],
        z: dict[tuple[int, str], float],
        baselines: dict[tuple[str, str], Baseline],
        threshold: float,
        monitored: tuple[str, ...],
        skipped: dict[str, int],
    ) -> None:
        self.anomalies = anomalies
        self.z = z
        self.baselines = baselines
        self.threshold = threshold
        self.monitored = monitored
        # 因为分层样本不足而**没有被判定**的步数。它必须与「正常」分开报。
        self.skipped = skipped

    def by_signal(self) -> dict[str, tuple[Anomaly, ...]]:
        out: dict[str, list[Anomaly]] = {}
        for a in self.anomalies:
            out.setdefault(a.signal, []).append(a)
        return {k: tuple(v) for k, v in out.items()}

    def at_step(self, step: int) -> tuple[Anomaly, ...]:
        return tuple(a for a in self.anomalies if a.step_lo <= step <= a.step_hi)

    def signals_at(self, step: int) -> tuple[str, ...]:
        return tuple(sorted({a.signal for a in self.at_step(step)}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomalies": [a.to_dict() for a in self.anomalies],
            "threshold": self.threshold,
            "monitored": list(self.monitored),
            "skipped_steps": dict(self.skipped),
            "baselines": [
                {"key": f"{sig}@{kind}", **b.to_dict()} for (sig, kind), b in sorted(self.baselines.items())
            ],
        }


def anomalies_of_signal(result: DetectResult, signal: str) -> tuple[Anomaly, ...]:
    return tuple(a for a in result.anomalies if a.signal == signal)


def _lookup(
    baselines: Mapping[tuple[str, str], Baseline],
    sig: str,
    kind: str,
    *,
    allow_fallback: bool,
) -> Baseline | None:
    """分层查表。

    `allow_fallback=False` 时**不拿全量基线兜底**：某个分层样本不足就如实不判定。
    拿全量基线去判一个样本不足的分层，正是「不分层」那个错误的翻版 ——
    本项目里它会把第 60 步（ckpt+eval 混合步）判成异常，并顺手把
    `vram_pressure` 的链污染成 `@60`。
    """
    b = baselines.get((sig, kind))
    if b is not None:
        return b
    return baselines.get((sig, _FALLBACK_KIND)) if allow_fallback else None


def z_table(
    step_table: Sequence[StepSignal],
    signals: Sequence[str],
    baselines: Mapping[tuple[str, str], Baseline],
    *,
    allow_fallback: bool = True,
) -> dict[tuple[int, str], float]:
    """`(步, 信号) → 有符号 z`。有符号，因为方向本身就是信息：
    GPU 利用率掉下去和涨上去是两件事，丢掉符号就分不出来了。"""
    out: dict[tuple[int, str], float] = {}
    for ss in step_table:
        kind = step_kind(ss)
        for sig in signals:
            v = ss.get(sig)
            if v is None:
                continue
            b = _lookup(baselines, sig, kind, allow_fallback=allow_fallback)
            if b is None:
                continue
            out[(ss.step, sig)] = b.z(v, ss.step)
    return out


def detect(step_table: Sequence[StepSignal], cfg: Mapping[str, Any]) -> DetectResult:
    dcfg = cfg["detect"]
    k = float(dcfg["k"])
    floor_ratio = float(dcfg["mad_floor_ratio"])
    min_size = int(dcfg["min_stratum_size"])
    min_episode = int(dcfg["min_episode_steps"])
    stratify = bool(dcfg["stratify_by_step_type"])
    detrend = bool(dcfg.get("detrend", True))
    span = float(dcfg.get("detrend_span", 4.0))
    fallback_mode = str(dcfg.get("stratum_fallback", "none"))
    monitored = tuple(str(s) for s in dcfg["monitored"])

    if not step_table:
        raise ValueError("逐步信号表是空的 —— 没有步，谈不上检出。")

    # 分层模式下**不兜底**：分层样本不足的步如实不判定。
    allow_fallback = not stratify or fallback_mode == "all"
    if stratify:
        baselines = stratified_baselines(
            step_table,
            monitored,
            floor_ratio,
            min_stratum_size=min_size,
            fallback_kind=_FALLBACK_KIND,
            detrend=detrend,
            span=span,
        )
    else:
        # 消融开关：**不分层**。留着它是为了让「分层有必要吗」这个问题
        # 能被实测回答，而不是靠信念。
        baselines = stratified_baselines(
            step_table,
            monitored,
            floor_ratio,
            min_stratum_size=1,
            fallback_kind=_FALLBACK_KIND,
            detrend=detrend,
            span=span,
        )
        baselines = {(sig, _FALLBACK_KIND): b for (sig, _kk), b in baselines.items() if _kk == _FALLBACK_KIND}

    zs = z_table(step_table, monitored, baselines, allow_fallback=allow_fallback)

    skipped: dict[str, int] = {sig: 0 for sig in monitored}
    raw: dict[str, list[tuple[int, float, float, bool]]] = {sig: [] for sig in monitored}
    for ss in step_table:
        kind = step_kind(ss)
        for sig in monitored:
            v = ss.get(sig)
            if v is None:
                continue
            if isinstance(v, float) and math.isnan(v):
                # 显式分支：NaN 一定异常，且不进 z 通道
                raw[sig].append((ss.step, float("nan"), float("nan"), True))
                continue
            z = zs.get((ss.step, sig))
            if z is None:
                # 没有基线 = **不判定**，不是「正常」。这两件事必须分开报，
                # 否则「未判定」会被读成「没问题」。
                skipped[sig] += 1
                continue
            bad = z if SIGNAL_HIGH_IS_BAD[sig] else -z
            if bad > k:
                raw[sig].append((ss.step, float(v), z, False))

    anomalies: list[Anomaly] = []
    for sig in monitored:
        hits = sorted(raw[sig], key=lambda t: t[0])
        for lo, hi, members in _episodes(hits):
            if len(members) < min_episode:
                continue
            nan_members = [m for m in members if m[3]]
            if nan_members:
                peak_step, peak_value, peak_z, is_nan = nan_members[0]
            else:
                peak_step, peak_value, peak_z, is_nan = max(members, key=lambda m: (abs(m[2]), -m[0]))
            anomalies.append(
                Anomaly(
                    id=f"an-{sig}-{lo:04d}",
                    signal=sig,
                    kind=WHAT_OF_SIGNAL[sig],
                    step_lo=lo,
                    step_hi=hi,
                    peak_step=peak_step,
                    peak_value=peak_value,
                    peak_z=peak_z,
                    is_nan=is_nan,
                )
            )

    anomalies.sort(key=lambda a: (a.step_lo, a.signal))
    return DetectResult(tuple(anomalies), zs, dict(baselines), k, monitored, skipped)


def _episodes(
    hits: Sequence[tuple[int, float, float, bool]],
) -> list[tuple[int, int, list[tuple[int, float, float, bool]]]]:
    """连续步聚成一段。**NaN 与 z 命中算同一类** —— 它们都表示「这一步不对劲」。"""
    out: list[tuple[int, int, list[tuple[int, float, float, bool]]]] = []
    cur: list[tuple[int, float, float, bool]] = []
    for h in hits:
        if cur and h[0] != cur[-1][0] + 1:
            out.append((cur[0][0], cur[-1][0], cur))
            cur = []
        cur.append(h)
    if cur:
        out.append((cur[0][0], cur[-1][0], cur))
    return out
