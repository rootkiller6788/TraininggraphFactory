"""第三方校验：**生成解释的人不给自己打分。**

`recover.py` 从事件流里复原出五问的答案；本模块把答案拿到**真值模拟器**上去试：

1. **α 规则（反事实充分性）**。在 `deps(D)` 上改动，若目标信号的变化小于
   `alpha_sigma × 噪音 σ`，那这个改动**不算充分条件** —— 一个「改变了 0.3%」的
   配置不叫根因。本项目把 `alpha_sigma` 取 **1.0**：变化不到一个噪音尺度，
   与噪音不可区分。（早先想过取 0.05，但那是**低于**噪音地板的门槛，
   等于什么都拦不住。）

2. **P / F / h 三个可判分探针**。只测「去掉它有没有变化」是不够的 ——
   一个「只要注了就都行」的判据也会全过。所以三个方向都要测：
   P 测充分性，F 测特异性（它声明**不**影响的信号必须不动），
   h 测定位性（把它挪走，原位置必须不动）。

3. **反事实反向链**。把某个原因从模拟器里**真的去掉**，重跑整条流水线，
   看它的归因是否消失。这是「驱动真实模拟器重放」那一句的执行点。

本模块**读真值**，因为它就是那个第三方。`recover.py` 一行真值都不读，
两者的分工不能混。
"""

from __future__ import annotations

import copy
import math
from typing import Any, Iterable, Mapping, Sequence

from .align import AlignmentError, align
from .config import causal_model_of, signatures_of
from .detect import detect
from .events import simulate
from .recover import group_key, recover
from .schema import (
    ROOT_CAUSE_KINDS,
    AlignmentAblation,
    AlphaCurve,
    Attribution,
    CounterfactualCheck,
    ProbeResult,
    StepSignal,
    VerifyResult,
)
from .stats import step_kind, stratified_baselines

__all__ = ["verify", "alpha_of", "F_SWEEP", "ALIGN_ABLATIONS"]

# 幅度扫描的网格。`f=0` 等价于移除，`f=1` 是原幅度。
F_SWEEP: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)

# 对齐消融的假设子集。顺序即报告顺序：从「什么漂移都不许用」到「只许常数偏移」。
# `identity` 单独那一档一定失败（残差等于偏移本身，第一步就出局）——
# 保留它正是为了把「连偏移都不估」这条路**当场演示成走不通**。
ALIGN_ABLATIONS: tuple[tuple[str, ...], ...] = (
    ("identity",),
    ("identity", "constant"),
)

# h 探针的诱饵与目标至少隔开这么多步。比任何异常窗口都大，
# 否则诱饵的物理会真的漫到目标位置上，那测的就不是「定位性」而是「重叠度」了。
_DECOY_GAP = 23


def _roots_of(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(r) for r in cfg["root_causes"]]


def _run_with(cfg: Mapping[str, Any], roots: Sequence[Mapping[str, Any]]) -> Any:
    """拿一份改了 `root_causes` 的配置重跑模拟器。**不落盘，纯计算。**"""
    c = dict(cfg)
    c["root_causes"] = [dict(r) for r in roots]
    return simulate(c)


def _scaled(root: Mapping[str, Any], f: float) -> dict[str, Any]:
    """`magnitude` 从 1（无效果）线性拉到原值。

    线性拉的是**偏离 1 的那部分**，不是乘数本身：`lr_spike` 的 mag=8 意思是
    「学习率 ×8」，f=0.5 应该得到 ×4.5（一半的偏离），不是 ×4。
    """
    r = dict(root)
    mag0 = float(r.get("magnitude", 1.0))
    r["magnitude"] = 1.0 + f * (mag0 - 1.0)
    return r


def _table(cfg: Mapping[str, Any], roots: Sequence[Mapping[str, Any]]) -> tuple[Any, tuple[StepSignal, ...]]:
    run = _run_with(cfg, roots)
    return run, align(run, cfg).step_table


def _value(table: Sequence[StepSignal], step: int, signal: str) -> float | None:
    for ss in table:
        if ss.step == step:
            return ss.get(signal)
    return None


def _transform(x: float | None, log_space: bool) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    if log_space:
        return math.log(x) if x > 0.0 else float("-inf")
    return float(x)


def _scale_of(
    table: Sequence[StepSignal],
    signal: str,
    step: int,
    cfg: Mapping[str, Any],
) -> tuple[float, bool]:
    """目标信号在该步所属分层上的噪音尺度（与检出共用同一套基线）。"""
    dcfg = cfg["detect"]
    bl = stratified_baselines(
        table,
        [signal],
        float(dcfg["mad_floor_ratio"]),
        min_stratum_size=int(dcfg["min_stratum_size"]),
        fallback_kind="all",
        detrend=bool(dcfg.get("detrend", True)),
        span=float(dcfg.get("detrend_span", 4.0)),
    )
    kind = "all"
    for ss in table:
        if ss.step == step:
            kind = step_kind(ss)
            break
    b = bl.get((signal, kind)) or bl.get((signal, "all"))
    if b is None:
        return 0.0, False
    return b.scale, b.log_space


def alpha_of(table: Sequence[StepSignal], signal: str, step: int, cfg: Mapping[str, Any]) -> tuple[float, float, bool]:
    """`(α, σ, log_space)`。α = `alpha_sigma × σ`。"""
    a_sigma = float(cfg["causal"].get("alpha_sigma", 1.0))
    sigma, log_space = _scale_of(table, signal, step, cfg)
    return a_sigma * sigma, sigma, log_space


def _delta(
    base: Sequence[StepSignal],
    other: Sequence[StepSignal],
    signal: str,
    step: int,
    log_space: bool,
) -> float:
    a = _value(base, step, signal)
    b = _value(other, step, signal)
    if a is None or b is None:
        return float("nan")
    ta, tb = _transform(a, log_space), _transform(b, log_space)
    if math.isnan(ta) or math.isnan(tb):
        # 一边是 NaN、另一边不是 —— 这是**开关式**的差别，不是量上的差别。
        # 用 inf 表示「不可能更充分了」。
        if math.isnan(ta) != math.isnan(tb):
            return float("inf")
        return float("nan")
    return abs(ta - tb)


def verify(
    cfg: Mapping[str, Any],
    base_table: Sequence[StepSignal],
    attributions: Sequence[Attribution],
) -> VerifyResult:
    model = causal_model_of(cfg)
    sigs = signatures_of(cfg)
    roots = _roots_of(cfg)
    a_sigma = float(cfg["causal"].get("alpha_sigma", 1.0))

    probes: list[ProbeResult] = []
    checks: list[CounterfactualCheck] = []
    curves: list[AlphaCurve] = []

    for a in attributions:
        key = group_key(a)
        kind = a.root_cause_kind
        spec = model[kind]
        signal = str(spec["primary_signal"])
        step = a.when_step

        # 找这次归因对应的是哪一个注入项（**第三方才做的事**：按种类与时刻对齐真值）
        match = None
        for r in roots:
            if str(r["kind"]) == kind and abs(int(r["at_step"]) - step) <= 2:
                match = r
                break
        if match is None:
            probes.append(
                ProbeResult(key, "P", signal, step, float("nan"), float("nan"), False, True, False, "真值里找不到对应注入项")
            )
            continue
        rid = str(match["id"])

        alpha, _sigma, log_space = alpha_of(base_table, signal, step, cfg)

        # ---- P：去掉它，目标信号要动 ----
        # 「去掉这个根因」的运行同时是 P 探针与反事实反向链的输入。两边各跑一遍
        # 模拟 + 对齐在结果上等价（模拟是确定性的），但平白多付一次整条流水线的成本。
        without = [r for r in roots if str(r["id"]) != rid]
        run_w = _run_with(cfg, without)
        al_w = align(run_w, cfg)
        t_without = al_w.step_table
        d_p = _delta(base_table, t_without, signal, step, log_space)
        suff_p = _suffices(d_p, alpha)
        probes.append(ProbeResult(key, "P", signal, step, d_p, alpha, suff_p, True, suff_p))

        # ---- F：去掉它，**它声明不影响的信号**必须不动 ----
        f_sig = _insensitive_signal(spec, model)
        if f_sig is None:
            probes.append(ProbeResult(key, "F", "", step, 0.0, alpha, False, False, True, "没有可用的不敏感信号"))
        else:
            alpha_f, _s2, log_f = alpha_of(base_table, f_sig, step, cfg)
            d_f = _delta(base_table, t_without, f_sig, step, log_f)
            suff_f = _suffices(d_f, alpha_f)
            probes.append(ProbeResult(key, "F", f_sig, step, d_f, alpha_f, suff_f, False, not suff_f))

        # ---- h：在别处注入一份**等量但不同种类**的诱饵，X 的目标信号必须不动 ----
        # 这条挡的是「Δ 到处都是」：如果连一个无关诱饵都能让 X 的目标信号动起来，
        # 那 P 探针测到的 Δ 就分不清是 X 造成的还是背景造成的。
        decoy_kind = _most_disjoint_kind(kind, sigs)
        decoy_step = _far_step(int(match["at_step"]), int(cfg["run"]["steps"]))
        decoy = {
            "id": "decoy",
            "kind": decoy_kind,
            "at_step": decoy_step,
            "duration_steps": int(match.get("duration_steps", 1)),
            "magnitude": float(match.get("magnitude", 1.0)),
        }
        _, t_decoy = _table(cfg, [*roots, decoy])
        d_h = _delta(base_table, t_decoy, signal, step, log_space)
        suff_h = _suffices(d_h, alpha)
        probes.append(
            ProbeResult(
                key,
                "h",
                signal,
                step,
                d_h,
                alpha,
                suff_h,
                False,
                not suff_h,
                f"诱饵 {decoy_kind}@{decoy_step}",
            )
        )

        # ---- 反事实反向链：去掉它，重跑整条流水线，看归因是否消失 ----
        # `run_w` / `t_without` 就是上面 P 探针那一次运行，直接复用。
        rec_w = recover(run_w, t_without, detect(t_without, cfg), cfg, al_w)
        same_kind = sum(1 for x in rec_w.attributions if x.root_cause_kind == kind)
        near = [x for x in rec_w.attributions if x.root_cause_kind == kind and abs(x.when_step - step) <= 2]
        checks.append(
            CounterfactualCheck(
                root_key=key,
                kind=kind,
                at_step=int(match["at_step"]),
                n_attributions_before=len(attributions),
                n_attributions_after=len(rec_w.attributions),
                chain_disappeared=not near,
                same_kind_after=same_kind,
            )
        )

        # ---- α 标定曲线 ----
        # 基线取**幅度压到 1（等价于没有效果）**的那一次运行，不是原运行 ——
        # 拿原运行当基线的话，曲线会从「满效果」走到 0，方向是反的，
        # 而我们要回答的是「幅度小到什么程度就不再是原因」。
        _, t_zero = _table(cfg, [(_scaled(r, 0.0) if str(r["id"]) == rid else dict(r)) for r in roots])
        points: list[tuple[float, float]] = []
        for f in F_SWEEP:
            scaled = [_scaled(r, f) if str(r["id"]) == rid else dict(r) for r in roots]
            _, t_f = _table(cfg, scaled)
            d = _delta(t_zero, t_f, signal, step, log_space)
            if math.isinf(d):
                ratio = float("inf")
            elif math.isnan(d) or not _sigma:
                ratio = 0.0
            else:
                ratio = d / _sigma
            points.append((f, ratio))
        f_star = -1.0
        for f, ratio in points:
            if ratio >= a_sigma:
                f_star = f
                break
        mono = all(points[i][1] <= points[i + 1][1] + 1e-9 for i in range(len(points) - 1))
        curves.append(AlphaCurve(key, kind, signal, tuple(points), f_star, mono))

    return VerifyResult(
        probes=tuple(probes),
        counterfactuals=tuple(checks),
        curves=tuple(curves),
        alpha_sigma=a_sigma,
        ablations=_alignment_ablations(cfg, roots, attributions),
        n_roots_recovered=_n_recovered(attributions, roots),
    )


def _n_recovered(attributions: Sequence[Attribution], roots: Sequence[Mapping[str, Any]]) -> int:
    """有几个注入项被归因**按种类 + 时刻**接住了。这是 recall 的分子。"""
    hit = 0
    for r in roots:
        kind = str(r["kind"])
        at = int(r["at_step"])
        for a in attributions:
            if a.root_cause_kind == kind and abs(a.when_step - at) <= 2:
                hit += 1
                break
    return hit


def _alignment_ablations(
    cfg: Mapping[str, Any],
    roots: Sequence[Mapping[str, Any]],
    attributions: Sequence[Attribution],
) -> tuple[AlignmentAblation, ...]:
    """把对齐假设锁进一个子集，**真的重跑整条流水线**，数还剩几条归因。

    这里要处理一个诚实的问题：`only=("identity",)` 时 `align` 会**抛错**而不是
    给出一个烂结果 —— 因为没有任何一种假设的残差落在容差内。抛错就是答案
    （「这条路走不通」），所以要把它记成一条 `error` 而不是让整个 `run` 崩掉。
    把失败记成 0 条而不是记成错误，会让报告看起来像「它跑出来了但很差」。
    """
    total = len(roots)
    out: list[AlignmentAblation] = []
    for only in ALIGN_ABLATIONS:
        try:
            run = simulate(cfg)
            al = align(run, cfg, only=only)
            table = al.step_table
            det = detect(table, cfg)
            rec = recover(run, table, det, cfg, al)
        except AlignmentError as exc:
            out.append(AlignmentAblation(only=tuple(only), n_roots_recovered=0, n_roots_total=total, error=str(exc)))
            continue
        out.append(
            AlignmentAblation(
                only=tuple(only),
                n_roots_recovered=_n_recovered(rec.attributions, roots),
                n_roots_total=total,
            )
        )
    return tuple(out)


def _suffices(delta: float, alpha: float) -> bool:
    if math.isnan(delta):
        return False
    if math.isinf(delta):
        return True
    return delta > alpha


# F 探针拿哪个信号当「该原因声明不影响」的对照。`net_mbps` 是刻意挑的：
# 它只受采样抖动影响，六个根因没有一个会碰它。于是「去掉 X 之后它也动了」
# 只可能是测量本身在乱动。
_INSENSITIVE = "net_mbps"


def _insensitive_signal(spec: Mapping[str, Any], model: Mapping[str, Mapping[str, Any]]) -> str | None:
    return _INSENSITIVE


def _most_disjoint_kind(kind: str, sigs: Mapping[str, Sequence[str]]) -> str:
    """与 `kind` 的签名重叠最少的那种 —— 拿它当诱饵最不容易「顺带」影响到目标信号。
    平手时按声明顺序取，保证决定论。"""
    mine = set(sigs[kind])
    best = kind
    best_overlap = None
    for k in ROOT_CAUSE_KINDS:
        if k == kind:
            continue
        ov = len(mine & set(sigs[k]))
        if best_overlap is None or ov < best_overlap:
            best_overlap = ov
            best = k
    return best


def _far_step(at_step: int, steps: int) -> int:
    """找一个离 `at_step` 至少 `_DECOY_GAP` 步、且落在 `[1, steps-2]` 内的位置。"""
    for cand in (at_step + _DECOY_GAP + 11, at_step - _DECOY_GAP - 11, steps // 2):
        if 1 <= cand <= steps - 2 and abs(cand - at_step) >= _DECOY_GAP:
            return cand
    return max(1, min(steps - 2, at_step + _DECOY_GAP + 11))
