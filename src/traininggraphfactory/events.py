"""训练模拟器与四源事件流。

**这是「上游」**：它按确定的物理规则跑一遍训练，把过程写成四路异质事件。
真实运行时里这四路来自四个不同的进程/库，各自有自己的时钟 —— 本模块把这件事
如实模拟出来：

| 来源 | 带 step 号？ | 它报告什么 |
| --- | --- | --- |
| `job` | **是**（唯一） | step 边界、ckpt/eval 边界、状态转移、配置变更、同步信标 |
| `metric` | 否 | loss / grad_norm / lr 采样 |
| `resource` | 否 | GPU / CPU / RAM / VRAM / Disk / Network 采样 |
| `io` | 否 | 读批、写 checkpoint、读 checkpoint、读评测集 |

**只有 `job` 带 step 号**，所以其余三路必须靠时钟对齐才能接到某一步上 —— 这就是
`align` 模块存在的全部理由。反过来，`job` 的偏移不可辨识（没有外部参照），
按定义取 0，于是所有偏移都是**相对 job 的偏移**。

三条刻意的设计：

1. **注入量写在配置里，不写代码**。时钟偏移、漂移率、根因位置与幅度全来自
   `configs/default.json`，因为流水线的任务就是把它们**测回来**。
2. **真值因果链是「跑出来之后记下来的」，不是配置里声明的**。配置只说「在第 42 步
   注入一个 lr_spike，幅度 8」。它到底带出了哪几条信号、在哪几步，是传播规则算出来的。
3. **`resource` 源在漂移**。常数偏移对它不够用 —— 这一点会由 `align` 报出来，
   并且会在验收里被量化成「漂移感知比常数偏移多救回多少条归因」。
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import causal_model_of, root_causes_of, sources_of
from .rng import Rng, derive_seed
from .schema import (
    FAILED_STATE_EDGE,
    Event,
    SourceSpec,
    StateTransition,
    StepTrace,
    TruthRecord,
    chain_of,
    evidence_hop,
    what_of_chain,
)

__all__ = [
    "SimRun",
    "simulate",
    "STATE_EDGES",
    "is_ckpt_step",
    "is_eval_step",
    "dump_events",
    "load_events",
]

# 声明的状态机转移图。**状态图里只有这些边** —— 出现别的转移就是事件流坏了，
# `state.py` 会拒绝它，而不是把图悄悄补一条边。
STATE_EDGES: tuple[tuple[str, str], ...] = (
    ("INIT", "PREPARING"),
    ("PREPARING", "TRAINING"),
    ("TRAINING", "EVALUATING"),
    ("EVALUATING", "TRAINING"),
    ("TRAINING", "CHECKPOINTING"),
    ("CHECKPOINTING", "TRAINING"),
    ("TRAINING", "PAUSED"),
    ("PAUSED", "TRAINING"),
    ("TRAINING", "FAILED"),
    ("FAILED", "RECOVER"),
    ("RECOVER", "TRAINING"),
    ("TRAINING", "DONE"),
)


def is_ckpt_step(step: int, ckpt_every: int) -> bool:
    return step > 0 and step % ckpt_every == 0


def is_eval_step(step: int, eval_every: int) -> bool:
    return step > 0 and step % eval_every == 0


@dataclass(frozen=True)
class SimRun:
    """模拟器的全部产出。`traces` 是**真时钟上的上游**，只用于审计与核对；
    其余四项才是「可以被分析的东西」。"""

    sources: tuple[SourceSpec, ...]
    traces: tuple[StepTrace, ...]
    events: tuple[Event, ...]
    truths: tuple[TruthRecord, ...]
    transitions: tuple[StateTransition, ...]
    config_timeline: tuple[tuple[int, str, Any], ...]
    beacons: tuple[tuple[int, int], ...]

    def events_of(self, source_id: str) -> tuple[Event, ...]:
        return tuple(e for e in self.events if e.source == source_id)

    def trace(self, step: int) -> StepTrace | None:
        for t in self.traces:
            if t.step == step:
                return t
        return None


# --------------------------------------------------------------------------- #
# 生效的根因 → 一组乘数
# --------------------------------------------------------------------------- #

_EFFECT_KEYS = (
    "lr_mult",
    "token_mult",
    "overhead_mult",
    "compute_mult",
    "vram_mult",
    "ckpt_mult",
    "disk_mult",
    "eval_mult",
    "grad_mult",
    "loss_mult",
)

# 每种根因的**参考幅度**：传播形状是在这个幅度上标定的。
#
# 注入幅度偏离它时，所有乘数按 `sev = ln(mag) / ln(ref)` 缩放 —— 于是幅度是个
# **真的物理参数**，而不是只写进 `config_value` 给人看的数字。
# 默认配置里每个根因的幅度都恰好等于它的参考值，所以 `sev = 1`，本表**不改变
# 任何默认行为**（默认运行逐位不变）。
_REF_MAG: dict[str, float] = {
    "lr_spike": 8.0,
    "bad_batch": 8.0,
    "vram_pressure": 1.5,
    "disk_stall": 5.0,
    "eval_regression": 1.9,
    "nan_grad": 1.0,
}


def _sev(mag: float, kind: str) -> float:
    """幅度的对数比例。`mag == ref` → 1.0；`mag == 1` → 0.0（无效果）。"""
    ref = _REF_MAG.get(kind, 1.0)
    if ref <= 1.0 or mag <= 1.0:
        return 1.0 if mag > 1.0 else 0.0
    return max(0.0, math.log(mag) / math.log(ref))


def _scale(base: float, sev: float) -> float:
    """把「乘数 base」按强度 `sev` 从 1 拉到 base。`sev=1` 时恒等于 base。"""
    return 1.0 + (base - 1.0) * sev


def _effects(step: int, roots: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    """第 `step` 步上，所有活跃根因叠出来的乘数。

    **`grad_mult` / `loss_mult` 是带时间形状的**，不是一个常数：学习率炸了之后
    梯度先炸、loss 后炸，这个**先后**正是因果链能被还原的原因。如果两者同时炸，
    「谁先动」这条唯一的因果线索就没有了。
    """
    out: dict[str, Any] = {k: 1.0 for k in _EFFECT_KEYS}
    out["nan"] = False
    out["active"] = []
    out["peak_label"] = {}

    for r in roots:
        at = int(r["at_step"])
        dur = int(r["duration_steps"])
        if not (at <= step < at + dur):
            continue
        mag = float(r["magnitude"])
        rel = step - at
        out["active"].append(str(r["id"]))
        kind = str(r["kind"])
        sev = _sev(mag, kind)

        if kind == "lr_spike":
            out["lr_mult"] *= mag
            grad_shape = {0: 18.0, 1: 8.0, 2: 2.6, 3: 1.4}.get(rel, 1.0)
            loss_shape = {0: 1.0, 1: 1.9, 2: 1.55, 3: 1.30, 4: 1.15}.get(rel, 1.0)
            out["grad_mult"] *= _scale(grad_shape, sev)
            out["loss_mult"] *= _scale(loss_shape, sev)
            if rel <= 2:
                # 梯度裁剪 + 回滚重试：host 侧多干活，GPU 就空转一点。
                out["overhead_mult"] *= _scale(1.28, sev)
            out["compute_mult"] *= _scale(1.04, sev)
            if grad_shape > out["peak_label"].get("grad_norm_z", (0.0, -1))[0]:
                out["peak_label"]["grad_norm_z"] = (grad_shape, step)
            if loss_shape > out["peak_label"].get("loss_z", (0.0, -1))[0]:
                out["peak_label"]["loss_z"] = (loss_shape, step)

        elif kind == "bad_batch":
            # 一批坏数据（超大 / 超长）：宿主侧的拼批与预处理被打爆，CPU 打满，
            # 于是 GPU 只能干等 —— **资源是 CPU，GPU 空转是后果不是现场**。
            out["token_mult"] *= mag
            out["overhead_mult"] *= _scale(10.0, sev)
            out["peak_label"]["batch_tokens_z"] = (mag, step)
            out["peak_label"]["step_time_z"] = (1.0, step)
            out["peak_label"]["cpu_z"] = (1.0, step)

        elif kind == "vram_pressure":
            out["vram_mult"] *= mag
            out["overhead_mult"] *= _scale(3.5, sev)
            out["compute_mult"] *= _scale(1.1, sev)
            out["peak_label"]["vram_z"] = (mag, step)
            out["peak_label"]["step_time_z"] = (1.0, step)
            out["peak_label"]["gpu_util_z_drop"] = (1.0, step)

        elif kind == "disk_stall":
            out["ckpt_mult"] *= mag
            out["disk_mult"] *= _scale(0.06, sev)
            out["peak_label"]["disk_z_drop"] = (1.0, step)
            out["peak_label"]["ckpt_write_z"] = (mag, step)
            out["peak_label"]["step_time_z"] = (1.0, step)

        elif kind == "eval_regression":
            # 评测集换了分片：读数变差（eval_loss 跳），而且**读得慢**（disk 吞吐掉）。
            out["eval_mult"] *= mag
            out["disk_mult"] *= _scale(0.4, sev)
            out["peak_label"]["eval_loss_z"] = (mag, step)
            out["peak_label"]["disk_z_drop"] = (1.0, step)
            # 注意：**这里故意不写 step_time_z**。声明的签名里有它，
            # 而真实传播不会让它出现 —— 这就是「声明了一条从没兑现的因果边」。

        elif kind == "nan_grad":
            # **开关式**：要么 nan 要么不 nan，没有中间量。所以幅度对它无意义
            # （配置里它的幅度就是 1.0），α 标定曲线对它必然退化。
            out["nan"] = True
            out["grad_mult"] *= 1e6
            out["peak_label"]["grad_norm_z"] = (1e6, step)
            out["peak_label"]["loss_nan"] = (1.0, step)
            out["peak_label"]["state_failed"] = (1.0, step)
            # `gpu_util_z_drop` 不在这里 —— 它出现在**下一步的恢复**里
            # （重载 checkpoint 的 1.4s 里 GPU 是空的），见 `_evidence_steps`。

    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def simulate(cfg: Mapping[str, Any]) -> SimRun:
    run = cfg["run"]
    steps = int(run["steps"])
    batch = int(run["batch_size"])
    seq_len = int(run["seq_len"])
    cv = float(run["length_cv"])
    ckpt_every = int(run["ckpt_every"])
    eval_every = int(run["eval_every"])
    base_lr = float(run["base_lr"])
    seed = int(cfg["seed"])

    roots = root_causes_of(cfg)
    model = causal_model_of(cfg)
    source_specs = tuple(SourceSpec.from_dict(s) for s in sources_of(cfg))
    # 事件流里每写一条事件都要查一次来源。`emit` 会被调用上千次，
    # 在这里做一次表，比每次 `next(...)` 扫一遍来源列表便宜得多。
    # 用 setdefault 而不是 dict 推导：与 `next` 一样取**第一个**同 id 的来源。
    spec_by_id: dict[str, SourceSpec] = {}
    for _spec in source_specs:
        spec_by_id.setdefault(_spec.id, _spec)
    beacon_every = int(cfg["beacon"]["every_steps"])
    beacon_jitter = float(cfg["beacon"]["jitter_s"])
    drop_prob = float(cfg["beacon"]["drop_prob"])

    # 初始配置。**它必须在事件流里**，否则「当时用的是哪份配置」无从回答。
    init_config: dict[str, Any] = {
        "train.lr": base_lr,
        "train.micro_batch": int(run["micro_batch"]),
        "train.grad_accum": int(run["grad_accum"]),
        "train.grad_clip": float(run["grad_clip"]),
        "data.max_seq_len": int(run["max_seq_len"]),
        "ckpt.every": ckpt_every,
        "data.eval_slice": "eval-v1",
        "run.precision": str(run["precision"]),
    }

    # 每个来源一条独立随机流：多注入一个根因不会改变已有根因的抖动。
    rngs = {s.id: Rng(derive_seed(seed, s.id, "events")) for s in source_specs}

    traces: list[StepTrace] = []
    events: list[Event] = []
    transitions: list[StateTransition] = []
    config_timeline: list[tuple[int, str, Any]] = [(0, k, v) for k, v in sorted(init_config.items())]

    ev_seq = 0

    def emit(source: str, kind: str, t_true: float, step: int | None, payload: dict[str, Any]) -> None:
        """把一条事件写到某个来源上，并把真时间换算成**该来源的本地时钟**。

        `t_local = t_true + offset + drift · t_true` —— 漂移是**对真时间**的线性项。
        注意这里不能用 `t_local` 自己当自变量，否则偏移与漂移不再线性可分，
        而 `align` 的线性拟合会给出一个系统性偏斜的斜率。
        """
        nonlocal ev_seq
        spec = spec_by_id[source]
        t_local = t_true + spec.clock_offset_s + spec.drift_s_per_s * t_true
        ev_seq += 1
        events.append(
            Event(
                id=f"e{ev_seq:06d}",
                source=source,
                kind=kind,
                t_local=t_local,
                step=(step if spec.carries_step else None),
                payload=payload,
            )
        )

    # ---- 状态机与逐步物理 ----
    state = "INIT"
    t_true = 0.0
    prev_lr = base_lr

    for s in range(steps):
        eff = _effects(s, roots)
        is_ckpt = is_ckpt_step(s, ckpt_every)
        is_eval = is_eval_step(s, eval_every)

        # 每个来源的采样抖动：按 (来源, 步) 派生，于是加一个根因不会改变抖动。
        jit = {sid: rngs[sid].normal(0.0, 0.004) for sid in rngs}

        # ---- 一批数据的长度（数据层的输入）
        lens = []
        for i in range(batch):
            z = rngs["job"].normal(0.0, 1.0)
            ln = seq_len * (1.0 + cv * z)
            lens.append(max(4, int(round(ln))))
        tokens = int(round(sum(lens) * eff["token_mult"]))
        tokens = min(tokens, int(run["max_seq_len"]) * batch * 8)

        # ---- 计算与开销
        compute_ms = (float(run["base_compute_ms"]) + float(run["per_token_ms"]) * tokens) * eff["compute_mult"]
        overhead_ms = float(run["overhead_ms"]) * eff["overhead_mult"]

        # ---- 状态转移（在一步之内发生）
        def go(frm: str, to: str, reason: str) -> None:
            nonlocal state
            if (frm, to) not in STATE_EDGES:
                raise ValueError(f"模拟器试图走一条未声明的转移：{frm} -> {to}")
            state = to
            transitions.append(StateTransition(step=s, frm=frm, to=to, t_local=t_true, reason=reason))
            emit("job", "state_change", t_true, s, {"frm": frm, "to": to, "reason": reason})

        if s == 0:
            go("INIT", "PREPARING", "run_begin")
        elif s == 1:
            go("PREPARING", "TRAINING", "data_ready")

        # 恢复步：上一步 nan 了（进了 FAILED），这一步从最近的 checkpoint 续跑。
        # **必须真的从 FAILED 出发** —— 从 TRAINING 直接「恢复」是一条没声明过的边，
        # `state.py` 会拒绝它。这条链正是 §8「异常链路关联」的主干。
        recover_step = any(
            int(r["at_step"]) + 1 == s and str(r["kind"]) == "nan_grad" for r in roots
        )
        recover_extra_ms = 0.0
        if recover_step:
            go("FAILED", "RECOVER", "recover_begin")
            go("RECOVER", "TRAINING", "resume_from_checkpoint")
            recover_extra_ms = float(run["recover_ms"])

        ckpt_extra_ms = 0.0
        eval_extra_ms = 0.0
        ckpt_id = ""
        eval_id = ""

        if state == "TRAINING":
            if is_eval:
                go("TRAINING", "EVALUATING", "eval_due")
                eval_extra_ms = float(run["eval_ms"])
                eval_id = f"eval-{s:04d}"
                go("EVALUATING", "TRAINING", "eval_done")
            if is_ckpt:
                go("TRAINING", "CHECKPOINTING", "ckpt_due")
                ckpt_extra_ms = float(run["ckpt_write_ms"]) * eff["ckpt_mult"]
                ckpt_id = f"ckpt-{s:04d}"
                go("CHECKPOINTING", "TRAINING", "ckpt_done")

        # 数值炸了 → 运行进入 FAILED。**这一步没有产出**：loss 是 nan，也不会写 ckpt。
        nan = bool(eff["nan"])
        if nan and state == "TRAINING":
            go("TRAINING", "FAILED", "nan_grad")

        step_time_ms = compute_ms + overhead_ms + ckpt_extra_ms + eval_extra_ms + recover_extra_ms
        step_time_s = step_time_ms / 1000.0
        t_begin = t_true
        t_end = t_true + step_time_s

        # ---- 信号值
        loss_trend = float(run["loss0"]) * (float(run["loss_decay"]) ** s)
        loss = loss_trend * eff["loss_mult"] * (1.0 + float(run["loss_noise"]) * rngs["metric"].normal(0.0, 1.0))
        grad_norm = (
            float(run["grad_norm_base"])
            * math.exp(float(run["grad_norm_sigma"]) * rngs["metric"].normal(0.0, 1.0))
            * eff["grad_mult"]
        )
        lr = base_lr * eff["lr_mult"]
        if nan:
            loss = float("nan")

        gpu0_util = 100.0 * compute_ms / step_time_ms
        gpu1_util = min(100.0, gpu0_util * 0.92 + 2.0)
        cpu_util = min(99.0, 34.0 * eff["overhead_mult"] ** 0.6 + 1.5 * rngs["resource"].normal(0.0, 1.0))
        ram_bytes = (16.0e9 + tokens * 9000.0) * (1.0 + 0.004 * rngs["resource"].normal(0.0, 1.0))
        vram_bytes = (
            float(run["params_m"]) * 1e6 * float(run["bytes_per_param"]) * (1.0 + float(run["vram_opt_mult"]))
        ) * eff["vram_mult"] + tokens * float(run["bytes_per_token_activation"])
        if ckpt_extra_ms > 0:
            disk_mbps = float(run["disk_mbps_write"]) * eff["disk_mult"]
        elif is_eval:
            disk_mbps = 60.0 * eff["disk_mult"]
        else:
            disk_mbps = 8.0 * eff["disk_mult"]
        net_mbps = float(run["net_mbps_base"]) * (1.0 + 0.03 * rngs["resource"].normal(0.0, 1.0))
        eval_loss = None
        if is_eval:
            eval_loss = (
                loss_trend * 0.92 * eff["eval_mult"] * (1.0 + 0.02 * rngs["metric"].normal(0.0, 1.0))
            )

        traces.append(
            StepTrace(
                step=s,
                t_true_begin=t_begin,
                t_true_end=t_end,
                tokens=tokens,
                compute_ms=compute_ms,
                overhead_ms=overhead_ms,
                loss=loss,
                grad_norm=grad_norm,
                lr=lr,
                gpu0_util=gpu0_util,
                gpu1_util=gpu1_util,
                cpu_util=cpu_util,
                ram_bytes=ram_bytes,
                vram_bytes=vram_bytes,
                disk_mbps=disk_mbps,
                net_mbps=net_mbps,
                is_ckpt=is_ckpt,
                is_eval=is_eval,
                ckpt_id=ckpt_id,
                eval_id=eval_id,
                eval_loss=eval_loss,
                nan=nan,
                root_causes=tuple(eff["active"]),
            )
        )

        # ---- 事件：job
        emit("job", "step_begin", t_begin + jit["job"], s, {"step": s, "lr": lr, "tokens": tokens})
        if is_eval:
            emit("job", "eval_begin", t_begin + jit["job"], s, {"eval_id": eval_id, "step": s})
            emit("job", "eval_end", t_end + jit["job"], s, {"eval_id": eval_id, "step": s})
        if is_ckpt:
            emit("job", "ckpt_begin", t_begin + jit["job"], s, {"ckpt_id": ckpt_id, "step": s})
            emit("job", "ckpt_end", t_end + jit["job"], s, {"ckpt_id": ckpt_id, "step": s})
        emit("job", "step_end", t_end + jit["job"], s, {"step": s})

        # ---- 事件：io
        # **放在步中点，不放在步首。** 步首正好压在步边界上，而步边界本身带 ±8ms
        # 抖动（job 的 `step_begin` 也有抖动）—— 两边的抖动一撞，采样就会掉进上一步。
        # 那是纯粹的测量噪声，不是「对齐做错了」，所以要在源头避开它。
        t_mid = (t_begin + t_end) / 2.0
        emit(
            "io",
            "read_batch",
            t_mid + jit["io"],
            None,
            {"tokens": tokens, "duration_s": overhead_ms / 1000.0},
        )
        if is_ckpt:
            emit(
                "io",
                "write_ckpt",
                t_mid + jit["io"],
                None,
                {
                    "ckpt_id": ckpt_id,
                    "bytes": float(run["params_m"]) * 1e6 * float(run["bytes_per_param"]),
                    "duration_s": ckpt_extra_ms / 1000.0,
                },
            )
        if recover_step:
            emit(
                "io",
                "read_ckpt",
                t_mid + jit["io"],
                None,
                {"ckpt_id": _checkpoint_at_or_before(s - 1, ckpt_every), "duration_s": recover_extra_ms / 1000.0},
            )
        if is_eval:
            emit("io", "read_eval", t_mid + jit["io"], None, {"eval_id": eval_id, "duration_s": eval_extra_ms / 1000.0})

        # ---- 配置变更（只在 lr 真的变了的那一步发）
        if abs(lr - prev_lr) > 1e-12:
            config_timeline.append((s, "train.lr", lr))
            emit("job", "config_change", t_begin + jit["job"], s, {"key": "train.lr", "value": lr})
            prev_lr = lr

        t_true = t_end

    # ---- 收尾。只有跑在 TRAINING 里才算正常收尾；卡在 FAILED 里的运行没有 DONE
    #      （`FAILED -> DONE` 这条边**没有声明**，所以这里不能硬补一条）。
    if state == "TRAINING":
        state = "DONE"
        transitions.append(
            StateTransition(step=steps - 1, frm="TRAINING", to="DONE", t_local=t_true, reason="run_complete")
        )
        emit("job", "state_change", t_true, steps - 1, {"frm": "TRAINING", "to": "DONE", "reason": "run_complete"})

    # ---- 事件：metric / resource 采样（按各自时钟的固定频率）
    # 采样点有几百个，而 traces 的起点按时间单调；把起点表建一次，归属就退化成二分。
    trace_starts = [tr.t_true_begin for tr in traces]
    for spec in source_specs:
        if spec.sample_hz is None:
            continue
        rng = rngs[spec.id]
        period = 1.0 / float(spec.sample_hz)
        n = int((t_true + spec.clock_offset_s) / period) + 1
        for k in range(n):
            t_local = k * period
            # 反解出真时间：t_local = t_true + off + drift·t_true
            t_sample = (t_local - spec.clock_offset_s) / (1.0 + spec.drift_s_per_s)
            if t_sample < 0.0 or t_sample > t_true:
                continue
            tr = _trace_at(traces, trace_starts, t_sample)
            if tr is None:
                continue
            if spec.kind == "metric":
                payload = {
                    "loss": tr.loss,
                    "grad_norm": tr.grad_norm,
                    "lr": tr.lr,
                    "tag": "train",
                    "nan": tr.nan,
                }
                emit(spec.id, "sample", t_sample, None, payload)
                if tr.is_eval and tr.eval_loss is not None:
                    emit(
                        spec.id,
                        "sample",
                        t_sample,
                        None,
                        {"eval_loss": tr.eval_loss, "tag": "eval", "eval_id": tr.eval_id},
                    )
            else:
                emit(
                    spec.id,
                    "sample",
                    t_sample,
                    None,
                    {
                        "gpu0_util": tr.gpu0_util,
                        "gpu1_util": tr.gpu1_util,
                        "cpu_util": tr.cpu_util,
                        "ram_bytes": tr.ram_bytes,
                        "vram_bytes": tr.vram_bytes,
                        "disk_mbps": tr.disk_mbps,
                        "net_mbps": tr.net_mbps,
                    },
                )

    # ---- 事件：同步信标（四个 sink 各记一条，用自己的本地时钟）
    beacons: list[tuple[int, int]] = []
    b_idx = 0
    for s in range(0, steps, beacon_every):
        tr = traces[s]
        beacons.append((b_idx, s))
        for spec in source_specs:
            if drop_prob > 0.0 and rngs[spec.id].bernoulli(drop_prob):
                continue
            jitter = rngs[spec.id].uniform(-beacon_jitter, beacon_jitter)
            t_true_b = tr.t_true_begin
            t_local = t_true_b + spec.clock_offset_s + spec.drift_s_per_s * t_true_b + jitter
            ev_seq += 1
            events.append(
                Event(
                    id=f"e{ev_seq:06d}",
                    source=spec.id,
                    kind="sync",
                    t_local=t_local,
                    step=(s if spec.carries_step else None),
                    payload={"beacon": b_idx},
                )
            )
        b_idx += 1

    # ---- run_begin / run_end（放在最后追加，顺序无关紧要 —— 分析一律按 kind 取）
    emit("job", "run_begin", 0.0, 0, {"config": dict(init_config), "steps": steps})
    emit("job", "run_end", t_true, steps - 1, {"steps": steps})

    # ---- 真值因果链
    truths = _build_truths(roots, model, traces, config_timeline, transitions, ckpt_every)

    events.sort(key=lambda e: (e.t_local, e.id))
    return SimRun(
        sources=source_specs,
        traces=tuple(traces),
        events=tuple(events),
        truths=truths,
        transitions=tuple(transitions),
        config_timeline=tuple(config_timeline),
        beacons=tuple(beacons),
    )


def _trace_at(traces: list[StepTrace], starts: list[float], t: float) -> StepTrace | None:
    """真时间 → 所属的步。步长可变（这正是异常造成的），所以必须**按实际边界查**，
    不能拿「平均步长」去除。

    `starts` 是 `traces` 的起点表（调用方建一次）。步起点按时间单调，于是这里
    用二分而不是线性扫描 —— 采样点有几百个，线性扫 120 步会让这一步变成
    `simulate` 里最费时的循环之一。语义与「逐条找包含区间」完全一致：
    最后一步之后的时间仍归最后一步，第一步之前的时间没有归属。
    """
    i = bisect.bisect_right(starts, t) - 1
    if i < 0:
        return None
    tr = traces[i]
    if t < tr.t_true_end or i == len(traces) - 1:
        return tr
    return None


def _checkpoint_at_or_before(step: int, ckpt_every: int) -> str:
    """异常时刻「最近的那个 checkpoint」= 恢复点。第 0 步之前没有 checkpoint，
    这时如实返回空串（而不是编一个 `ckpt@0`）。"""
    k = (step // ckpt_every) * ckpt_every
    return f"ckpt-{k:04d}" if k > 0 else ""


def _build_truths(
    roots: tuple[Mapping[str, Any], ...],
    model: Mapping[str, Mapping[str, Any]],
    traces: list[StepTrace],
    config_timeline: list[tuple[int, str, Any]],
    transitions: list[StateTransition],
    ckpt_every: int,
) -> tuple[TruthRecord, ...]:
    """把「注入 + 传播」翻译成一条**可判分的真值链**。

    两个约定是刻意与复原器对齐的（见 `recover.py`）：

    1. `chain` 的每一步记的是「该证据**首次出现**在哪一步」（不是峰值步）。
       取首次而不是峰值，是因为一个持续 3 步的等价跌幅里「哪一步是峰」是个平局 ——
       让真值和复原器都去猜平局，链的比对就变成在比运气。
    2. 键名由 `evidence_hop()` 生成，两边共用同一个函数。

    约定不一致的话，「链对不对」这个指标就在测量格式，而不是在测量归因能力。
    """
    out: list[TruthRecord] = []
    for r in roots:
        rid = str(r["id"])
        kind = str(r["kind"])
        at = int(r["at_step"])
        spec = model[kind]
        sig = tuple(str(x) for x in spec["signature"])

        steps = _evidence_steps(kind, at)
        present = [(lab, steps[lab]) for lab in sig if lab in steps]

        cfg_key = str(spec["config_key"])
        ckpt_key = _checkpoint_at_or_before(at, ckpt_every)
        triples = [
            (
                *evidence_hop(
                    lab,
                    st,
                    config_key=cfg_key,
                    state_key=FAILED_STATE_EDGE,
                    ckpt_key=ckpt_key,
                ),
                st,
            )
            for lab, st in present
        ]

        # 「什么时候」= 该根因**主信号**的峰值步。lr_spike 的主信号 loss 比注入晚一步
        # （梯度先炸、loss 后炸），所以这一问的答案不是 `at_step`。
        primary = str(spec["primary_signal"])
        when = at + _PEAK_OFFSET.get(kind, 0)

        # 配置在「什么时候」那一步的生效值。**从时间线里读，不是从根因声明里读** ——
        # 根因里根本没有 config_key（那是 causal_model 的字段），这是刻意的：
        # 让复原器没法从注入项里抄答案。
        cfg_value: Any = None
        for st, k, v in config_timeline:
            if k == cfg_key and st <= when:
                cfg_value = v

        out.append(
            TruthRecord(
                root_cause_id=rid,
                kind=kind,
                at_step=at,
                duration_steps=int(r["duration_steps"]),
                magnitude=float(r["magnitude"]),
                what=_what_of(primary, kind),
                when_step=when,
                resource=str(spec["resource"]),
                config_key=cfg_key,
                config_value=cfg_value,
                checkpoint=_checkpoint_at_or_before(when, ckpt_every),
                chain=chain_of(triples),
                recovery=str(spec.get("recovery", "")),
            )
        )
    return tuple(sorted(out, key=lambda t: (t.when_step, t.root_cause_id)))


# 每种根因的主信号峰值相对 `at_step` 的偏移。
# **只有 lr_spike 不是 0** —— 因为它注入的是学习率（config 层），而它被判分的主信号是
# loss（metric 层），中间隔着一步梯度爆炸。
_PEAK_OFFSET: dict[str, int] = {
    "lr_spike": 1,
    "bad_batch": 0,
    "vram_pressure": 0,
    "disk_stall": 0,
    "eval_regression": 0,
    "nan_grad": 0,
}


def _evidence_steps(kind: str, at: int) -> dict[str, int]:
    """每种根因在物理上真正会兑现的证据标签 → **首次出现**的步。

    这个约定与复原器严格一致（见 `recover.py` 的「记录每条证据的首次满足步」）。
    取首次而不是峰值：`disk_stall` 在第 72/73/74 步造成**等比例**的吞吐跌幅，
    「哪一步是峰」是个平局，让两边各自去猜平局就等于让链的比对变成比运气。

    两个刻意为之的缺席：

    - `eval_regression` 的 `step_time_z` 不在这里 —— 声明的签名里有它，
      传播规则里没有。这条「从没兑现的因果边」是 `expectations` 里
      `declared_signatures_all_materialize` 被推翻的全部原因。
    - `nan_grad` 的 `gpu_util_z_drop` 落在 `at + 1`：第 `at` 步只是**判死**，
      真正的 GPU 空转发生在下一步重载 checkpoint 的 1.4s 里。因果链因此
      跨过了步边界 —— 这正是「异常 → 资源」那条边不能靠同步发生来识别的原因。
    """
    if kind == "lr_spike":
        return {"config_change": at, "grad_norm_z": at, "loss_z": at + 1}
    if kind == "bad_batch":
        return {"batch_tokens_z": at, "step_time_z": at, "cpu_z": at}
    if kind == "vram_pressure":
        return {"vram_z": at, "step_time_z": at, "gpu_util_z_drop": at}
    if kind == "disk_stall":
        return {"disk_z_drop": at, "ckpt_write_z": at, "step_time_z": at}
    if kind == "eval_regression":
        return {"eval_loss_z": at, "disk_z_drop": at}
    if kind == "nan_grad":
        return {
            "grad_norm_z": at,
            "loss_nan": at,
            "state_failed": at,
            "gpu_util_z_drop": at + 1,
            "ckpt_restore": at + 1,
        }
    raise ValueError(f"未声明的根因种类：{kind!r}")


def _what_of(primary: str, kind: str) -> str:
    """与 `recover.py` 共用同一条规则（见 `schema.what_of_chain`）——
    两边各写一遍的话，「发生了什么」这一问就会变成在比格式。"""
    labels = ("loss_nan",) if kind == "nan_grad" else ()
    return what_of_chain(primary, labels)


# --------------------------------------------------------------------------- #
# 事件流的采集与回读（M0：事件流骨架）
# --------------------------------------------------------------------------- #


def dump_events(events: Iterable[Event], path: Path | str) -> Path:
    """把事件流落成 JSONL。**一行一个事件，按 (t_local, id) 排序。**

    排序不是为了好看：四路来源各自按自己的本地时钟产出，合在一起时
    「谁先谁后」是个没有绝对答案的问题（时钟本来就不同步）。按本地时间排，
    等于用「没对齐的钟」去决定顺序 —— 所以这里的顺序**只用于稳定 diff**，
    不承载任何语义。要顺序，去问 `align`。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(events, key=lambda e: (e.t_local, e.id))
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for e in rows:
            fh.write(json.dumps(e.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def load_events(path: Path | str) -> tuple[Event, ...]:
    """回读事件流。**行序即文件序**，不重排 —— 回读的意义是「拿到当初写下的东西」。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"没有事件流文件：{path}")
    out: list[Event] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        out.append(Event.from_dict(json.loads(ln)))
    return tuple(out)
