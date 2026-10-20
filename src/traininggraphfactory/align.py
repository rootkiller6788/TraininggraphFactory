"""时钟对齐与漂移检测。

**为什么这个模块存在**：四路事件里只有 `job` 带 step 号。其余三路（`metric` /
`resource` / `io`）只给本地时间戳，而它们的时钟和 `job` 不一致 —— 所以「把一条采样
接到哪一步上」不是读出来的，是**估出来的**。估错了，后面三张图和整条因果链全错。

三个必须能被数据推翻的问题：

1. **对齐有必要吗？** —— `identity`（不对齐）的归属错误率是多少。
2. **常数偏移够吗？** —— `constant`（只减一个中位残差）的残差在容差内吗。
3. **漂移要建模吗？** —— `linear`（最小二乘 `t_local = a + b·t_ref`）的斜率
   相对 1 的偏离是否显著。

三种假设**同台竞技**，判据是**步归属错误率**而不是拟合优度 —— 拟合得再好，
接错了步也没有意义。

两条不可越界的纪律：

- 估计**只用同步信标**，不用时间戳。信标有全局序号，序号对应关系是确定的；
  时间戳不是（它就是要被校正的东西）。
- 注入的时钟参数**只用于给对齐打分**，绝不参与估计。这跟 `TruthRecord` 与
  `Attribution` 分成两个文件是同一条纪律：**生成解释的人不给自己打分**。
"""

from __future__ import annotations

import bisect
import math
from typing import Any, Iterable, Mapping, Sequence

from .schema import (
    SIGNALS,
    AlignmentModel,
    AlignmentReport,
    AlignmentResult,
    Event,
    SourceSpec,
    StepSignal,
    StepTrace,
)

__all__ = [
    "AlignmentError",
    "MODELS",
    "ALIGNED_SIGNALS",
    "beacon_pairs",
    "step_boundaries",
    "fit_models",
    "align",
    "assign_step",
]

# 从简到繁。选择规则在打平时偏向靠前的那个 —— 多出来的参数必须**挣得**自己的位置。
MODELS: tuple[str, ...] = ("identity", "constant", "linear")

# 需要靠对齐才能接到步上的信号：`(来源, 事件类型, payload 键, 信号名)`。
# 写成一张表而不是一堆 if，是为了让「哪些信号是估出来的」一眼可见 ——
# 剩下那几个（step_time_s / throughput / lr）来自 `job`，本来就带 step 号。
ALIGNED_SIGNALS: tuple[tuple[str, str, str, str], ...] = (
    ("metric", "sample", "loss", "loss"),
    ("metric", "sample", "grad_norm", "grad_norm"),
    ("metric", "sample", "eval_loss", "eval_loss"),
    ("resource", "sample", "gpu0_util", "gpu0_util"),
    ("resource", "sample", "gpu1_util", "gpu1_util"),
    ("resource", "sample", "cpu_util", "cpu_util"),
    ("resource", "sample", "ram_bytes", "ram_bytes"),
    ("resource", "sample", "vram_bytes", "vram_bytes"),
    ("resource", "sample", "disk_mbps", "disk_mbps"),
    ("resource", "sample", "net_mbps", "net_mbps"),
    ("io", "read_batch", "tokens", "batch_tokens"),
    ("io", "write_ckpt", "duration_s", "ckpt_write_s"),
)

# 这几个来自 `job` 自己的事件，**不需要对齐**（它们本来就带 step 号）。
JOB_SIGNALS: tuple[str, ...] = ("step_time_s", "throughput", "lr")

_REFERENCE_SOURCE = "job"


class AlignmentError(RuntimeError):
    """对齐失败。**宁可不给答案，也不给一个悄悄退回 0 偏移的答案。**"""


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #


def _median(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError("median of empty")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _rms(residuals: Sequence[float]) -> float:
    if not residuals:
        return 0.0
    return math.sqrt(sum(r * r for r in residuals) / len(residuals))


def _least_squares(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """`y = a + b·x` 的最小二乘解。`x` 是参考时钟，`y` 是本来源的本地时钟。

    注意自变量必须是**参考时钟**，不能是本来源自己的时间戳 —— 否则偏移与漂移
    不再线性可分，斜率会被系统性地拉偏。
    """
    n = len(xs)
    if n < 2:
        return (ys[0] if ys else 0.0), 1.0
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    if abs(den) < 1e-12:
        b = 1.0
    else:
        b = (n * sxy - sx * sy) / den
    a = (sy - b * sx) / n
    return a, b


# --------------------------------------------------------------------------- #
# 信标与步边界
# --------------------------------------------------------------------------- #


def beacon_pairs(
    run: Any,
    source_id: str,
    reference_id: str = _REFERENCE_SOURCE,
) -> list[tuple[float, float]]:
    """`(本来源本地时间, 参考来源本地时间)` —— 只收**两边都记到**的信标。

    配对靠的是信标的**全局序号**（payload 里的 `beacon`），不是时间戳。
    参考时钟按定义是 `job` 自己的本地时钟：它带 step 号，是唯一的锚。
    """
    ref: dict[int, float] = {}
    for e in run.events:
        if e.kind == "sync" and e.source == reference_id:
            ref[int(e.payload["beacon"])] = e.t_local
    out: list[tuple[float, float]] = []
    for e in run.events:
        if e.kind == "sync" and e.source == source_id:
            b = int(e.payload["beacon"])
            if b in ref:
                out.append((e.t_local, ref[b]))
    out.sort(key=lambda p: p[1])
    return out


def step_boundaries(
    run: Any,
    reference_id: str = _REFERENCE_SOURCE,
) -> list[tuple[int, float, float]]:
    """`[(step, t_ref_begin, t_ref_end)]`，时间一律在**参考时钟**上。

    来自 `job` 的 `step_begin` / `step_end` 事件。同一步的这两条事件用的是**同一个**
    抖动抽样，所以两者相减得到的步长是精确的 —— 抖动在差分里抵消掉了。
    """
    begin: dict[int, float] = {}
    end: dict[int, float] = {}
    for e in run.events:
        if e.source != reference_id or e.step is None:
            continue
        if e.kind == "step_begin":
            begin[e.step] = e.t_local
        elif e.kind == "step_end":
            end[e.step] = e.t_local
    steps = sorted(set(begin) & set(end))
    return [(s, begin[s], end[s]) for s in steps]


def assign_step(
    t_ref: float,
    starts: Sequence[float],
    bounds: Sequence[tuple[int, float, float]],
) -> int | None:
    """把参考时钟上的一个时刻接到某一步上。

    用 `bisect` 查**真实的步边界表**，不是拿平均步长去除 —— 步长本身是被异常改变
    的东西（recover 步比训练步长 10 倍），用平均值会在异常附近系统性地接错。
    """
    i = bisect.bisect_right(starts, t_ref) - 1
    if i < 0:
        return None
    step, b0, b1 = bounds[i]
    if t_ref < b0 or t_ref >= b1:
        return None
    return step


# --------------------------------------------------------------------------- #
# 拟合
# --------------------------------------------------------------------------- #


def fit_models(pairs: Sequence[tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """三种假设各自的 `(a, b)`。模型形式：`t_local = a + b · t_ref`。"""
    ys = [p[0] for p in pairs]
    xs = [p[1] for p in pairs]

    lin_a, lin_b = _least_squares(xs, ys)
    const_a = _median([y - x for x, y in zip(xs, ys)])
    return {
        "identity": (0.0, 1.0),
        "constant": (const_a, 1.0),
        "linear": (lin_a, lin_b),
    }


def _residuals(pairs: Sequence[tuple[float, float]], a: float, b: float) -> list[float]:
    return [y - (a + b * x) for y, x in pairs]


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def align(run: Any, cfg: Mapping[str, Any], *, only: Sequence[str] | None = None) -> AlignmentResult:
    """把四路异质事件压成一张逐步信号表，并留下三种对齐假设的对比。

    `run.traces`（真时钟）**只用来给对齐打分**：把采样按注入的时钟参数反解回真时间，
    查它真正属于哪一步，再和估出来的步比。真值不参与任何估计。

    `only` 把可选的假设**限制在一个子集**里。这不是调试开关，是校验要用的：
    「漂移感知比常数偏移多救回多少条归因」这个问题只能靠**真的把 linear 拿掉重跑**
    来回答，不能靠看残差表猜。传 `only` 时若子集里没有一种假设够格，
    仍然按同样的规则报错 —— 不许悄悄放宽。
    """
    allowed = tuple(MODELS) if only is None else tuple(only)
    for name in allowed:
        if name not in MODELS:
            raise AlignmentError(f"未声明的对齐假设：{name!r}（可选：{', '.join(MODELS)}）")
    if not allowed:
        raise AlignmentError("`only` 不能为空 —— 一种假设都不许试，就没得选了。")

    acfg = cfg["align"]
    residual_tol = float(acfg["residual_tol_s"])
    drift_tol = float(acfg["drift_tol_s_per_s"])
    min_beacons = int(acfg["min_beacons"])
    max_err = float(acfg["max_step_assign_error"])

    specs = {s.id: s for s in run.sources}
    bounds = step_boundaries(run)
    if not bounds:
        raise AlignmentError("`job` 没有给出任何步边界 —— 没有锚，谈不上对齐。")
    starts = [b[1] for b in bounds]
    step_ids = [b[0] for b in bounds]

    # 真时钟的步边界，只用于打分
    truth_starts = [t.t_true_begin for t in run.traces]
    truth_steps = [t.step for t in run.traces]

    def truth_step_of(e: Event) -> int | None:
        sp = specs[e.source]
        t_true = (e.t_local - sp.clock_offset_s) / (1.0 + sp.drift_s_per_s)
        i = bisect.bisect_right(truth_starts, t_true) - 1
        if i < 0:
            return None
        return truth_steps[i]

    # 待归属的事件：不带 step 号、又不是信标的那些
    pending = [e for e in run.events if e.source != _REFERENCE_SOURCE and e.kind != "sync"]

    reports: list[AlignmentReport] = []
    candidates: list[tuple[str, tuple[AlignmentModel, ...]]] = []
    chosen: dict[str, str] = {}
    chosen_fit: dict[str, tuple[float, float]] = {}

    for sid in sorted(specs):
        sp = specs[sid]
        if sid == _REFERENCE_SOURCE:
            # 参考源按定义偏移为 0、不漂移。**这不是假设，是坐标系的选取**。
            chosen[sid] = "identity"
            chosen_fit[sid] = (0.0, 1.0)
            reports.append(
                AlignmentReport(
                    source=sid,
                    n_beacons=sum(1 for e in run.events if e.source == sid and e.kind == "sync"),
                    model="identity",
                    injected_offset_s=sp.clock_offset_s,
                    estimated_offset_s=0.0,
                    injected_drift_s_per_s=sp.drift_s_per_s,
                    estimated_drift_s_per_s=0.0,
                    residual_rms_s=0.0,
                )
            )
            candidates.append((sid, (AlignmentModel("identity", 0.0, 1.0, 0.0),)))
            continue

        pairs = beacon_pairs(run, sid)
        if len(pairs) < min_beacons:
            raise AlignmentError(
                f"来源 {sid!r} 只配到 {len(pairs)} 个信标（要求 ≥ {min_beacons}）—— "
                "信标不足时估出来的偏移不可信，这里不猜。"
            )

        fits = fit_models(pairs)
        mine = [e for e in pending if e.source == sid]

        models: list[AlignmentModel] = []
        for name in allowed:
            a, b = fits[name]
            resid = _residuals(pairs, a, b)
            n_ok = 0
            n_bad = 0
            n_wrong = 0
            for e in mine:
                st = assign_step((e.t_local - a) / b, starts, bounds)
                if st is None:
                    n_bad += 1
                    continue
                n_ok += 1
                ts = truth_step_of(e)
                if ts is not None and ts != st:
                    n_wrong += 1
            models.append(
                AlignmentModel(
                    model=name,
                    a=a,
                    b=b,
                    residual_rms_s=_rms(resid),
                    n_assigned=n_ok,
                    n_unassigned=n_bad,
                    assign_error_rate=(n_wrong / n_ok) if n_ok else 1.0,
                )
            )

        # 选择规则，两步：
        #   ① 残差必须落在容差内 —— 拟合不上信标的模型直接出局（`identity` 就是这样
        #      出局的：它的残差等于偏移本身）。
        #   ② 在剩下的里面挑归属错误率最低的；打平则取更简单的模型。
        #      `linear` 因此必须**挣得**它的第二个参数：斜率不显著就别要。
        eligible = [m for m in models if m.residual_rms_s <= residual_tol]
        if not eligible:
            best = min(models, key=lambda m: m.residual_rms_s)
            raise AlignmentError(
                f"来源 {sid!r} 没有一种假设的残差在 {residual_tol}s 容差内"
                f"（最好的是 {best.model}：{best.residual_rms_s:.4f}s）。"
                "时钟行为不只是偏移 + 漂移，这里不猜。"
            )
        if eligible[0].model == "linear" and abs(eligible[0].b - 1.0) <= drift_tol:
            # 斜率不显著 ⇒ 退回常数偏移，不白拿一个参数
            eligible = [m for m in eligible if m.model != "linear"] or eligible
        eligible.sort(key=lambda m: (m.assign_error_rate, MODELS.index(m.model)))
        win = eligible[0]

        if win.assign_error_rate > max_err:
            raise AlignmentError(
                f"来源 {sid!r} 用 {win.model} 对齐后仍有 {win.assign_error_rate:.1%} 的采样"
                f"接错步（上限 {max_err:.1%}）—— 后面的因果链会建在错位的信号上。"
            )

        chosen[sid] = win.model
        chosen_fit[sid] = (win.a, win.b)
        candidates.append((sid, tuple(models)))
        reports.append(
            AlignmentReport(
                source=sid,
                n_beacons=len(pairs),
                model=win.model,
                injected_offset_s=sp.clock_offset_s,
                estimated_offset_s=win.a,
                injected_drift_s_per_s=sp.drift_s_per_s,
                estimated_drift_s_per_s=win.drift_s_per_s,
                residual_rms_s=win.residual_rms_s,
                n_assigned=win.n_assigned,
                n_unassigned=win.n_unassigned,
            )
        )

    table, n_assigned = _build_step_table(run, step_ids, bounds, starts, chosen_fit, pending)
    return AlignmentResult(
        reports=tuple(reports),
        candidates=tuple(candidates),
        chosen_model=dict(chosen),
        step_table=table,
        step_boundaries=tuple(bounds),
        n_events_total=len(pending),
        n_events_assigned=n_assigned,
    )


def _build_step_table(
    run: Any,
    step_ids: Sequence[int],
    bounds: Sequence[tuple[int, float, float]],
    starts: Sequence[float],
    fits: Mapping[str, tuple[float, float]],
    pending: Iterable[Event],
) -> tuple[tuple[StepSignal, ...], int]:
    """逐步信号表：**对齐之后的公共输入**，三张图与检出都读它。"""
    by_step: dict[int, dict[str, list[float]]] = {s: {} for s in step_ids}
    contrib: dict[int, dict[str, int]] = {s: {} for s in step_ids}
    n_assigned = 0

    for e in pending:
        a, b = fits[e.source]
        st = assign_step((e.t_local - a) / b, starts, bounds)
        if st is None:
            continue
        n_assigned += 1
        for src, kind, key, sig in ALIGNED_SIGNALS:
            if e.source != src or e.kind != kind:
                continue
            v = e.payload.get(key)
            if v is None:
                continue
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            if isinstance(v, float) and math.isnan(v):
                # NaN 要**留在表里**：它不是「没测到」，它是「测到了，值就是 nan」。
                # 检出模块会显式判它，而不是靠容差比较 —— 拿 nan 去比大小永远是 False。
                by_step[st].setdefault(sig, []).append(float("nan"))
            else:
                by_step[st].setdefault(sig, []).append(float(v))
            contrib[st][e.source] = contrib[st].get(e.source, 0) + 1

    # `job` 自带的三个信号：不需要对齐，直接从带 step 号的事件里读
    job_vals: dict[int, dict[str, float]] = {s: {} for s in step_ids}
    for e in run.events:
        if e.source != _REFERENCE_SOURCE or e.step is None or e.step not in job_vals:
            continue
        if e.kind == "step_begin":
            job_vals[e.step]["lr"] = float(e.payload["lr"])
        elif e.kind == "step_end":
            pass
    for step, b0, b1 in bounds:
        job_vals[step]["step_time_s"] = b1 - b0

    table: list[StepSignal] = []
    for step, b0, b1 in bounds:
        values: dict[str, float | None] = {name: None for name in SIGNALS}
        n_samples = 0
        for sig, vs in by_step[step].items():
            values[sig] = _mean(vs)
            n_samples += len(vs)
        for sig in JOB_SIGNALS:
            if sig in job_vals[step]:
                values[sig] = job_vals[step][sig]
        tokens = values.get("batch_tokens")
        st_s = values.get("step_time_s")
        values["throughput"] = (tokens / st_s) if (tokens is not None and st_s) else None
        table.append(
            StepSignal(
                step=step,
                t_ref_begin=b0,
                values=values,
                n_samples=n_samples,
                n_sources=len(contrib[step]),
                assigned_from=dict(sorted(contrib[step].items())),
            )
        )
    return tuple(table), n_assigned


def _mean(xs: Sequence[float]) -> float:
    """步内聚合用均值。**只要有一条是 nan，这一步就是 nan** ——
    把 nan 平均掉会得到一个看起来很正常的数，那比没有数更糟。"""
    if any(math.isnan(x) for x in xs):
        return float("nan")
    return sum(xs) / len(xs)
