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

# `step_time_s` / `lr` 来自 `job` 自己的事件（**不需要对齐**，它们本来就带
# step 号）；`throughput` 由 `batch_tokens / step_time_s` 在建表时派生。


def _aligned_index() -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
    """`(来源, 事件类型) → ((payload 键, 信号名), ...)`。

    逐条事件扫整张 `ALIGNED_SIGNALS` 是这里最容易被忽略的开销：每次对齐要过
    上千条事件，而一条事件只可能命中其中一种 `(来源, 类型)`。先把表翻成索引，
    命中判断就退化成一次 dict 查找 —— 结果与线性扫描逐位相同。
    """
    out: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for src, kind, key, sig in ALIGNED_SIGNALS:
        out.setdefault((src, kind), []).append((key, sig))
    return {k: tuple(v) for k, v in out.items()}


_ALIGNED_BY_EVENT = _aligned_index()

# 同一张索引再按来源拆一层：`来源 → {事件类型: 信号表}`。逐事件热路径上
# 原来每条事件都要现拼一个 `(source, kind)` 元组去查表；拆层之后是先按
# 来源取子表（四个来源），再按 kind 取，省掉元组分配与哈希。
_ALIGNED_BY_SOURCE: dict[str, dict[str, tuple[tuple[str, str], ...]]] = {}
for _key, _sigs in _ALIGNED_BY_EVENT.items():
    _ALIGNED_BY_SOURCE.setdefault(_key[0], {})[_key[1]] = _sigs
_EMPTY_SIGNALS: dict[str, tuple[tuple[str, str], ...]] = {}

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


def _beacon_pairs(
    ref_sync: Iterable[Event],
    mine_sync: Iterable[Event],
) -> list[tuple[float, float]]:
    """信标配对的实体：两边各自的本来源信标**已经按事件流顺序**分好。"""
    ref: dict[int, float] = {}
    for e in ref_sync:
        ref[int(e.payload["beacon"])] = e.t_local
    out: list[tuple[float, float]] = []
    for e in mine_sync:
        b = int(e.payload["beacon"])
        if b in ref:
            out.append((e.t_local, ref[b]))
    out.sort(key=lambda p: p[1])
    return out


def beacon_pairs(
    run: Any,
    source_id: str,
    reference_id: str = _REFERENCE_SOURCE,
) -> list[tuple[float, float]]:
    """`(本来源本地时间, 参考来源本地时间)` —— 只收**两边都记到**的信标。

    配对靠的是信标的**全局序号**（payload 里的 `beacon`），不是时间戳。
    参考时钟按定义是 `job` 自己的本地时钟：它带 step 号，是唯一的锚。
    """
    ref_sync = [e for e in run.events if e.kind == "sync" and e.source == reference_id]
    mine_sync = [e for e in run.events if e.kind == "sync" and e.source == source_id]
    return _beacon_pairs(ref_sync, mine_sync)


def _bounds_from(
    begin: Mapping[int, float],
    end: Mapping[int, float],
) -> list[tuple[int, float, float]]:
    steps = sorted(set(begin) & set(end))
    return [(s, begin[s], end[s]) for s in steps]


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
    return _bounds_from(begin, end)


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

    # 一次遍历把后面要反复用的四样东西分好：
    #   begin/end       —— 参考源的步边界（下面合成 `bounds`）；
    #   sync_by_source  —— 每路来源的信标（信标配对要按来源取）；
    #   pending         —— 待归属事件（保持事件流顺序，建表要用它）；
    #   pending_by_source —— 同一批事件按来源分好（模型循环不再逐个筛）。
    # 原来这四样分别由 `step_boundaries`、三次 `beacon_pairs`、一次列表推导
    # 和每来源一次过滤得到，同一份 1500+ 条事件流要扫七八遍。
    begin: dict[int, float] = {}
    end: dict[int, float] = {}
    lr_by_step: dict[int, float] = {}
    sync_by_source: dict[str, list[Event]] = {}
    pending: list[Event] = []
    pending_by_source: dict[str, list[Event]] = {}
    for e in run.events:
        if e.kind == "sync":
            sync_by_source.setdefault(e.source, []).append(e)
            continue
        if e.source == _REFERENCE_SOURCE:
            if e.step is not None:
                if e.kind == "step_begin":
                    begin[e.step] = e.t_local
                    # `job` 自带 lr。建表时要用，这一趟本来就在扫全部事件，
                    # 顺手收下 —— 建表时再走一遍事件流纯属重复。
                    lr_by_step[e.step] = float(e.payload["lr"])
                elif e.kind == "step_end":
                    end[e.step] = e.t_local
            continue
        pending.append(e)
        pending_by_source.setdefault(e.source, []).append(e)

    bounds = _bounds_from(begin, end)
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

    # 真值只用于**打分**，而且与「试哪种模型」无关。它在模型循环里被反复调用
    # （每种假设都会把每一条事件判一次），所以在这里一次算清 —— 否则三种假设
    # 会把同一件事做三遍。真值仍然不参与估计。
    #
    # 按来源各建一条与 `pending_by_source` **位置对齐**的列表：模型循环里
    # 直接按下标并排取。原来建的是 `id(event) → 真值` 的全局表，模型循环
    # 再把每条事件的 `id()` 查回来 —— 一次对齐要白做两千多次 `id()` 与字典存取。
    truth_by_source: dict[str, list[int | None]] = {
        sid: [truth_step_of(e) for e in evs] for sid, evs in pending_by_source.items()
    }

    reports: list[AlignmentReport] = []
    candidates: list[tuple[str, tuple[AlignmentModel, ...]]] = []
    chosen: dict[str, str] = {}
    # 每种假设下逐事件算出的步号。胜出的那种直接交给建表用 —— 建表时再判
    # 一遍是纯粹的重复计算（同样的 a、b、starts、bounds，结果必然相同）。
    # 键是 (来源, 假设名)，值只收「判出了步」的事件（id → 步号）。
    assigned_by_model: dict[tuple[str, str], dict[int, int]] = {}

    # 步归属是逐事件的热路径（每次对齐要判上千次，而且每种假设都要判一遍）。
    # `assign_step` 的逻辑本身只有一个二分 + 两个区间检查，函数调用开销与
    # 二分本身同量级 —— 在这个循环里就地展开，判定与返回值逐位不变。
    bisect_right = bisect.bisect_right

    for sid in sorted(specs):
        sp = specs[sid]
        if sid == _REFERENCE_SOURCE:
            # 参考源按定义偏移为 0、不漂移。**这不是假设，是坐标系的选取**。
            chosen[sid] = "identity"
            reports.append(
                AlignmentReport(
                    source=sid,
                    n_beacons=len(sync_by_source.get(sid, ())),
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

        pairs = _beacon_pairs(
            sync_by_source.get(_REFERENCE_SOURCE, ()),
            sync_by_source.get(sid, ()),
        )
        if len(pairs) < min_beacons:
            raise AlignmentError(
                f"来源 {sid!r} 只配到 {len(pairs)} 个信标（要求 ≥ {min_beacons}）—— "
                "信标不足时估出来的偏移不可信，这里不猜。"
            )

        fits = fit_models(pairs)
        mine = pending_by_source.get(sid, [])
        # 真值步号只与事件有关，与试哪种模型无关，已按来源位置排好。
        truth_by_idx = truth_by_source.get(sid, ())

        models: list[AlignmentModel] = []
        assigned_lists: dict[str, list[int | None]] = {}
        for name in allowed:
            a, b = fits[name]
            resid = _residuals(pairs, a, b)
            n_ok = 0
            n_bad = 0
            n_wrong = 0
            # 先按事件位置收「判出的步」（判不出记 None）。归属结果只有
            # **胜出**的那一种假设要交给建表，没必要每种假设都建一个 id 字典。
            steps_i: list[int | None] = []
            append = steps_i.append
            # `zip` 把「事件」与「真值步」并排推进，省掉 enumerate 的计数与下标取值。
            for e, ts in zip(mine, truth_by_idx):
                t_ref = (e.t_local - a) / b
                i = bisect_right(starts, t_ref) - 1
                if i < 0:
                    append(None)
                    n_bad += 1
                    continue
                assigned, b0, b1 = bounds[i]
                if t_ref < b0 or t_ref >= b1:
                    append(None)
                    n_bad += 1
                    continue
                n_ok += 1
                append(assigned)
                if ts is not None and ts != assigned:
                    n_wrong += 1
            assigned_lists[name] = steps_i
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
        # 胜出的假设才需要 `id(event) → 步` 的映射，落选的那些归属结果
        # 只用于打分，不必为每条事件建一个键。
        assigned_by_model[(sid, win.model)] = {
            id(e): st for e, st in zip(mine, assigned_lists[win.model]) if st is not None
        }
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

    # 参考源没有「待归属事件」（`pending` 已经排除它），不需要映射。
    winner_steps = {
        sid: assigned_by_model[(sid, name)] for sid, name in chosen.items() if sid != _REFERENCE_SOURCE
    }
    table, n_assigned = _build_step_table(step_ids, bounds, winner_steps, pending, lr_by_step)
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
    step_ids: Sequence[int],
    bounds: Sequence[tuple[int, float, float]],
    steps_by_source: Mapping[str, Mapping[int, int]],
    pending: Iterable[Event],
    lr_by_step: Mapping[int, float],
) -> tuple[tuple[StepSignal, ...], int]:
    """逐步信号表：**对齐之后的公共输入**，三张图与检出都读它。

    `steps_by_source` 是胜出假设下逐事件的步号（`id(event) → step`）。
    没在表里的事件就是「判不出步」，计数上算未归属。
    """
    by_step: dict[int, dict[str, list[float]]] = {s: {} for s in step_ids}
    contrib: dict[int, dict[str, int]] = {s: {} for s in step_ids}
    aligned_by_source = _ALIGNED_BY_SOURCE
    n_assigned = 0

    for e in pending:
        src = e.source
        st = steps_by_source[src].get(id(e))
        if st is None:
            continue
        n_assigned += 1
        signals = aligned_by_source.get(src, _EMPTY_SIGNALS).get(e.kind)
        if not signals:
            continue
        step_vals = by_step[st]
        payload_get = e.payload.get
        n_vals = 0
        for key, sig in signals:
            v = payload_get(key)
            if v is None:
                continue
            # 载荷绝大多数是 float，先判它、再退回 int 的子类判定 ——
            # 判定结果与 `isinstance(v, (int, float)) and not isinstance(v, bool)` 完全一致。
            if isinstance(v, float):
                # NaN 要**留在表里**：它不是「没测到」，它是「测到了，值就是 nan」。
                # 检出模块会显式判它，而不是靠容差比较 —— 拿 nan 去比大小永远是 False。
                # `v != v` 是 NaN 的定义本身，省掉一次 `math.isnan` 调用；非 NaN
                # 的 float 原值存下即可（`float(v)` 对 float 是恒等）。
                x = float("nan") if v != v else v
            elif isinstance(v, int) and not isinstance(v, bool):
                x = float(v)
            else:
                continue
            bucket = step_vals.get(sig)
            if bucket is None:
                bucket = step_vals[sig] = []
            bucket.append(x)
            n_vals += 1
        if n_vals:
            # 计数按「来源贡献了几个数值」累加，不必每个值都去查一次来源表。
            src_counts = contrib[st]
            src_counts[src] = src_counts.get(src, 0) + n_vals

    table: list[StepSignal] = []
    for step, b0, b1 in bounds:
        values: dict[str, float | None] = dict.fromkeys(SIGNALS)
        n_samples = 0
        for sig, vs in by_step[step].items():
            # 步内聚合用均值。**只要有一条是 nan，这一步就是 nan** —— 把 nan
            # 平均掉会得到一个看起来很正常的数，那比没有数更糟。
            #
            # 判 NaN 不必先扫一遍生成器：只要有一条是 nan，`sum` 的结果就一定
            # 是 nan（nan 沿加法传播）。先求和再判，省掉生成器构造与一次全表
            # 遍历；`sum` 本身不动，因此逐位结果与原来完全一致。
            total = sum(vs)
            values[sig] = float("nan") if math.isnan(total) else total / len(vs)
            n_samples += len(vs)
        # `job` 自带的信号：步耗时就是步边界之差；lr 由 `align` 的索引遍历
        # 顺路收好（那次本来就要扫全部事件），这里不必再走一遍事件流。
        values["step_time_s"] = b1 - b0
        lr = lr_by_step.get(step)
        if lr is not None:
            values["lr"] = lr
        tokens = values.get("batch_tokens")
        st_s = values.get("step_time_s")
        values["throughput"] = (tokens / st_s) if (tokens is not None and st_s) else None
        # 位置传参：每步一条，字段顺序就是 `StepSignal` 的契约。
        table.append(
            StepSignal(
                step,
                b0,
                values,
                n_samples,
                len(contrib[step]),
                dict(sorted(contrib[step].items())),
            )
        )
    return tuple(table), n_assigned
