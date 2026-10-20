"""**第三张图：Resource Graph。** `Job → {GPU0, GPU1, CPU, RAM, VRAM, Disk, Network}`。

它回答的问题不是「用了多少」，而是「**这一步压在哪个资源上**」。三个设计要点：

1. **必须归一化。** VRAM 的量纲是几十亿字节，GPU 利用率是百分比。不归一化的话，
   峰值永远落在 VRAM 上 —— 一个永远不变的答案等于没有答案。这里用**该资源自己
   的稳健 z** 做归一化，于是七个资源第一次可比。
2. **方向必须写对。** GPU 利用率**掉下来**才是异常（在空转），磁盘吞吐**掉下来**
   才是异常；CPU / RAM / VRAM **涨上去**才是异常。方向写错不会报错，
   只会让「瓶颈在哪个资源」整个反掉。
3. **门槛与检出共用。** 资源图说「瓶颈」的门槛和检出说「异常」的门槛是同一个
   `detect.k`。两处各定一个门槛，报告里就会出现「瓶颈是 CPU，但 CPU 没有异常」
   这种没人能解释的话。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .jobdag import GraphError
from .schema import (
    RESOURCES,
    RESOURCE_HIGH_IS_BAD,
    RESOURCE_SIGNAL,
    ResourceGraph,
    ResourcePressure,
    StepSignal,
)
from .stats import Baseline, step_kind, stratified_baselines

__all__ = ["HUB", "build_resource_graph", "signal_baselines"]

HUB = "Job"


def signal_baselines(step_table: Sequence[StepSignal], cfg: Mapping[str, Any]) -> dict[tuple[str, str], Baseline]:
    """七路资源信号各自的**分层**基线。检出模块也读这一张表。"""
    dcfg = cfg["detect"]
    floor_ratio = float(dcfg["mad_floor_ratio"])
    min_size = int(dcfg["min_stratum_size"])
    signals = [RESOURCE_SIGNAL[r] for r in RESOURCES]
    return stratified_baselines(
        step_table,
        signals,
        floor_ratio,
        min_stratum_size=min_size,
        fallback_kind="all",
        detrend=bool(dcfg.get("detrend", True)),
        span=float(dcfg.get("detrend_span", 4.0)),
    )


def _baseline_for(
    table: Mapping[tuple[str, str], Baseline],
    signal: str,
    kind: str,
) -> Baseline:
    b = table.get((signal, kind)) or table.get((signal, "all"))
    if b is None:
        raise GraphError(f"信号 {signal!r} 在 {kind!r} 层上既没有分层基线也没有全量基线。")
    return b


def build_resource_graph(step_table: Sequence[StepSignal], cfg: Mapping[str, Any]) -> ResourceGraph:
    if not step_table:
        raise GraphError("逐步信号表是空的 —— 没有步，谈不上资源图。")
    table = signal_baselines(step_table, cfg)
    k = float(cfg["detect"]["k"])

    # 每个资源在每一步的「坏方向 z」。正数 = 该资源这一步比平时更受压。
    bad_z: dict[str, dict[int, float]] = {r: {} for r in RESOURCES}
    for ss in step_table:
        kind = step_kind(ss)
        for r in RESOURCES:
            sig = RESOURCE_SIGNAL[r]
            v = ss.get(sig)
            if v is None:
                continue
            b = _baseline_for(table, sig, kind)
            z = b.z(v, ss.step)
            bad_z[r][ss.step] = z if RESOURCE_HIGH_IS_BAD[r] else -z

    pressure: list[ResourcePressure] = []
    for r in RESOURCES:
        zs = bad_z[r]
        if not zs:
            continue
        peak_step = max(sorted(zs), key=lambda s: (zs[s], -s))
        sig = RESOURCE_SIGNAL[r]
        ss = step_table[peak_step]
        v = ss.get(sig)
        b = _baseline_for(table, sig, step_kind(ss))
        pressure.append(
            ResourcePressure(
                resource=r,
                peak_step=peak_step,
                peak_value=float(v) if v is not None else float("nan"),
                baseline=b.expected(ss.step),
                normalized=zs[peak_step],
                direction="high" if RESOURCE_HIGH_IS_BAD[r] else "low",
            )
        )

    # 每一步的瓶颈 = 坏方向 z 最大的那个资源；**没超过门槛就报空**。
    # 报一个「这一步的瓶颈是 GPU1，z = 1.4」等于报噪声 —— 不如如实说没有。
    bottleneck: list[tuple[int, str]] = []
    for ss in step_table:
        best_r = ""
        best_z = k
        for r in RESOURCES:
            z = bad_z[r].get(ss.step)
            if z is None:
                continue
            # 平手时按 RESOURCES 的声明顺序取 —— 决定论
            if z > best_z:
                best_z = z
                best_r = r
        bottleneck.append((ss.step, best_r))

    return ResourceGraph(
        hub=HUB,
        resources=RESOURCES,
        signal_of=dict(RESOURCE_SIGNAL),
        edges=tuple((HUB, r) for r in RESOURCES),
        pressure=tuple(pressure),
        bottleneck_by_step=tuple(bottleneck),
        n_steps=len(step_table),
    )
