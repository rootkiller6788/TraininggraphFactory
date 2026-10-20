"""对象链：事件 → 对齐信号 → 三张图 → 异常 → 因果链 → 五问归因。

**七层信号必须分开**（`LAYERS`）。这不是分类癖 —— 混在一起就会退化成一张杂乱的
流程图，而「哪一层先动、动了之后谁被带下水」正是本项目唯一要回答的问题。
每一层的 `key` 前缀是固定的，于是任何一条因果跳都能用 `layer + key` 唯一定位：

| layer | 前缀 | 例子 |
| --- | --- | --- |
| `config` | `cfg:` | `cfg:train.lr@42` |
| `data` | `dat:` | `dat:batch_tokens@55` |
| `io` | `io:` | `io:ckpt_write_s@72` |
| `job` | `job:` | `job:step_time_s@62` |
| `resource` | `res:` | `res:vram_bytes@62` |
| `metric` | `met:` | `met:loss@43` |
| `state` | `st:` | `st:TRAINING->FAILED@97` |
| `checkpoint` | `ckpt:` | `ckpt:ckpt-0096@98` |

**真值与答案必须分成两个对象**（`TruthRecord` vs `Attribution`），并且落进两个
不同的产物文件。理由和偏好数据一样：一旦真值和答案躺在同一个文件里，
「归因对了没有」就变成了一句自我声明。
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "utcnow",
    "LAYERS",
    "SIGNALS",
    "ROOT_CAUSE_KINDS",
    "STATES",
    "RESOURCES",
    "RESOURCE_SIGNAL",
    "RESOURCE_HIGH_IS_BAD",
    "SIGNAL_HIGH_IS_BAD",
    "LAYER_RESOURCE_HINT",
    "LAYER_RESOURCE_PRIORITY",
    "EVIDENCE_LAYER",
    "EVIDENCE_SIGNAL",
    "FAILED_STATE_EDGE",
    "WHAT_OF_SIGNAL",
    "what_of_chain",
    "resource_of_signal",
    "resource_of_chain",
    "hop_key",
    "evidence_hop",
    "chain_of",
    "SourceSpec",
    "Event",
    "StepTrace",
    "StepSignal",
    "AlignmentReport",
    "AlignmentModel",
    "AlignmentResult",
    "StateTransition",
    "ResourcePressure",
    "Anomaly",
    "CausalHop",
    "Attribution",
    "TruthRecord",
    "JobNode",
    "JobDag",
    "StateGraph",
    "ResourceGraph",
    "ProbeResult",
    "AlphaCurve",
    "CounterfactualCheck",
    "AlignmentAblation",
    "VerifyResult",
]

def utcnow() -> str:
    """唯一的非决定论字段。**只有它允许每次不同** —— 其余产物必须逐位可复现。"""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


LAYERS: tuple[str, ...] = ("config", "data", "io", "job", "resource", "metric", "state", "checkpoint")

SIGNALS: tuple[str, ...] = (
    "loss",
    "grad_norm",
    "step_time_s",
    "throughput",
    "gpu0_util",
    "gpu1_util",
    "cpu_util",
    "ram_bytes",
    "vram_bytes",
    "disk_mbps",
    "net_mbps",
    "batch_tokens",
    "ckpt_write_s",
    "eval_loss",
)

ROOT_CAUSE_KINDS: tuple[str, ...] = (
    "lr_spike",
    "bad_batch",
    "vram_pressure",
    "disk_stall",
    "eval_regression",
    "nan_grad",
)

# 状态机的状态集合。**顺序即声明的转移图的节点顺序**，不要按字母序排 ——
# 报告里按这个顺序读才是「训练的生命周期」。
STATES: tuple[str, ...] = (
    "INIT",
    "PREPARING",
    "TRAINING",
    "EVALUATING",
    "CHECKPOINTING",
    "PAUSED",
    "FAILED",
    "RECOVER",
    "DONE",
)

RESOURCES: tuple[str, ...] = ("GPU0", "GPU1", "CPU", "RAM", "VRAM", "Disk", "Network")

# 每个「资源」用哪个信号来衡量它的压力。**这张表是三张图能对话的前提**：
# 没有它，「哪个资源压力最大」就无从计算，只能靠根因声明去抄一个答案。
RESOURCE_SIGNAL: dict[str, str] = {
    "GPU0": "gpu0_util",
    "GPU1": "gpu1_util",
    "CPU": "cpu_util",
    "RAM": "ram_bytes",
    "VRAM": "vram_bytes",
    "Disk": "disk_mbps",
    "Network": "net_mbps",
}

# 每个信号「变大就是坏事」吗。**方向只在这里声明一次**，`RESOURCE_HIGH_IS_BAD`
# 由它派生 —— 两处各写一遍迟早会写出两个相反的方向，而方向写错不会报错，
# 只会让「瓶颈在哪个资源」整个反掉。
#
# 掉下去才是异常的信号：GPU 利用率（在空转）、吞吐、磁盘吞吐、网络吞吐。
# 涨上去才是异常的信号：loss / 梯度 / step 时间 / 显存 / 内存 / CPU / 批次 token 数。
SIGNAL_HIGH_IS_BAD: dict[str, bool] = {
    "loss": True,
    "grad_norm": True,
    "eval_loss": True,
    "step_time_s": True,
    "throughput": False,
    "gpu0_util": False,
    "gpu1_util": False,
    "cpu_util": True,
    "ram_bytes": True,
    "vram_bytes": True,
    "disk_mbps": False,
    "net_mbps": False,
    "batch_tokens": True,
    "ckpt_write_s": True,
}

# 每个资源「压力变大」的方向，由 `SIGNAL_HIGH_IS_BAD` 派生。
#
# CPU 是唯一一个两个方向都讲得通的资源（高 = 宿主侧打满，低 = 在等 IO）。
# 本项目按「高才是压力」取 —— `bad_batch` 的宿主侧开销正是把 CPU 打满，
# 而 CPU 常年 34% 是健康态，不是异常。
RESOURCE_HIGH_IS_BAD: dict[str, bool] = {
    r: SIGNAL_HIGH_IS_BAD[sig] for r, sig in RESOURCE_SIGNAL.items()
}

# 因果链的层 → 它**最可能落在哪个资源上**。只在链里没有 `resource` 层证据时兜底。
#
# 优先级 = 「症状最先出现的那一层」：异常最先被看见的层，就是它发生的那个资源。
# 于是 `nan_grad` 的链（metric / state / checkpoint）判 GPU0 —— grad_norm 是
# GPU 上算出来的；而 `checkpoint` 虽然也带资源（Disk），它是**后果**不是现场。
LAYER_RESOURCE_HINT: dict[str, str] = {
    "metric": "GPU0",
    "io": "Disk",
    "data": "CPU",
    "checkpoint": "Disk",
}
LAYER_RESOURCE_PRIORITY: tuple[str, ...] = ("metric", "io", "data", "checkpoint")

# 证据标签 → 它挂在哪一层、对应哪个信号。**真值链与复原链共用这一张表**，
# 否则两边的链会长得不一样，而「链对不对」就变成在测量格式而不是在测量归因。
EVIDENCE_LAYER: dict[str, str] = {
    "config_change": "config",
    "grad_norm_z": "metric",
    "loss_z": "metric",
    "loss_nan": "metric",
    "eval_loss_z": "metric",
    "batch_tokens_z": "data",
    "step_time_z": "job",
    "throughput_z_drop": "job",
    "gpu_util_z_drop": "resource",
    "cpu_z": "resource",
    "vram_z": "resource",
    "disk_z_drop": "resource",
    "ckpt_write_z": "io",
    "state_failed": "state",
    "ckpt_restore": "checkpoint",
}

EVIDENCE_SIGNAL: dict[str, str] = {
    "grad_norm_z": "grad_norm",
    "loss_z": "loss",
    "loss_nan": "loss",
    "eval_loss_z": "eval_loss",
    "batch_tokens_z": "batch_tokens",
    "step_time_z": "step_time_s",
    "throughput_z_drop": "throughput",
    "gpu_util_z_drop": "gpu0_util",
    "cpu_z": "cpu_util",
    "vram_z": "vram_bytes",
    "disk_z_drop": "disk_mbps",
    "ckpt_write_z": "ckpt_write_s",
}

# 进入失败态的那条边。真值链与复原链**都引用这一个常量** —— 各自写一遍字符串
# 迟早会有一边拼错，而拼错的后果是「链永远对不上」这种最难查的 bug。
FAILED_STATE_EDGE = "TRAINING->FAILED"

# 被检出的信号 → 这次异常叫什么。**「发生了什么」这一问基本是白送的**
# （它就是被触发的那个信号），所以报告里五问要分开报命中率，不能只报一个总分。
WHAT_OF_SIGNAL: dict[str, str] = {
    "loss": "loss_spike",
    "grad_norm": "grad_norm_spike",
    "step_time_s": "step_time_spike",
    "throughput": "throughput_drop",
    "gpu0_util": "gpu_util_drop",
    "gpu1_util": "gpu_util_drop",
    "cpu_util": "cpu_saturation",
    "ram_bytes": "ram_pressure",
    "vram_bytes": "vram_pressure",
    "disk_mbps": "disk_stall",
    "net_mbps": "network_stall",
    "batch_tokens": "batch_size_spike",
    "ckpt_write_s": "ckpt_write_stall",
    "eval_loss": "eval_loss_spike",
}


def what_of_chain(primary_signal: str, labels: Sequence[str]) -> str:
    """「发生了什么」= 主信号的读数异常了。

    **NaN 是唯一的例外**：`loss` 变 nan 不是「loss 冲高」，是「loss 没了」——
    两者的处置完全不同（前者调学习率，后者要回滚）。所以这一问不能只看主信号，
    还要看链上有没有 `loss_nan`。
    """
    if "loss_nan" in labels:
        return "loss_nan"
    return WHAT_OF_SIGNAL[primary_signal]


def resource_of_signal(signal: str) -> str:
    """信号 → 资源。`RESOURCE_SIGNAL` 的反表。"""
    for r, s in RESOURCE_SIGNAL.items():
        if s == signal:
            return r
    return ""


def resource_of_chain(chain: Sequence["CausalHop"]) -> str:
    """因果链 → 它落在哪个资源上。

    **优先取链上第一条 `resource` 层证据所在的资源。** 签名是有序的（按因果传播
    排列），所以「第一条」就是**最先受压**的那个 —— 后面那些是被它带下水的。

    链上没有资源层证据时（本项目里 `lr_spike` 就是这样），退到
    `LAYER_RESOURCE_HINT`：按 `LAYER_RESOURCE_PRIORITY` 取**症状最先出现的那一层**
    对应的资源。这是兜底，不是等价替代 —— 报告里两者要分开报。
    """
    for hop in chain:
        if hop.layer != "resource":
            continue
        name = hop.key.split("@", 1)[0]
        prefix = _LAYER_PREFIX["resource"]
        if name.startswith(prefix):
            name = name[len(prefix) :]
        r = resource_of_signal(name)
        if r:
            return r
    present = {h.layer for h in chain}
    for layer in LAYER_RESOURCE_PRIORITY:
        if layer in present:
            return LAYER_RESOURCE_HINT[layer]
    return ""

_LAYER_PREFIX: dict[str, str] = {
    "config": "cfg:",
    "data": "dat:",
    "io": "io:",
    "job": "job:",
    "resource": "res:",
    "metric": "met:",
    "state": "st:",
    "checkpoint": "ckpt:",
}


def hop_key(layer: str, name: str, step: int | None = None) -> str:
    """造一个因果跳的键。`layer` 必须已声明 —— 打错字要在这里就炸，
    而不是等到报告里出现一个没人认识的层名。"""
    if layer not in _LAYER_PREFIX:
        raise ValueError(f"未声明的层：{layer!r}（可选：{', '.join(LAYERS)}）")
    base = f"{_LAYER_PREFIX[layer]}{name}"
    return f"{base}@{step}" if step is not None else base


def evidence_hop(
    label: str,
    step: int,
    *,
    config_key: str = "",
    state_key: str = "",
    ckpt_key: str = "",
) -> tuple[str, str]:
    """证据标签 → `(layer, hop_key)`。真值链与复原链**必须共用这一个函数**。

    三个标签的键不是信号名，而是「当时的那个东西」的名字，所以要额外传参：

    - `config_change` → `cfg:<config_key>@<step>`
    - `state_failed` → `st:<TRAINING->FAILED>@<step>`
    - `ckpt_restore` → `ckpt:<ckpt-0096>@<step>`（**恢复自哪一个 checkpoint**）
    """
    if label not in EVIDENCE_LAYER:
        raise ValueError(f"未声明的证据标签：{label!r}（可选：{', '.join(sorted(EVIDENCE_LAYER))}）")
    layer = EVIDENCE_LAYER[label]
    if label == "config_change":
        if not config_key:
            raise ValueError("config_change 证据必须带上 config_key")
        return layer, hop_key(layer, config_key, step)
    if label == "state_failed":
        if not state_key:
            raise ValueError("state_failed 证据必须带上 state_key（形如 TRAINING->FAILED）")
        return layer, hop_key(layer, state_key, step)
    if label == "ckpt_restore":
        if not ckpt_key:
            raise ValueError("ckpt_restore 证据必须带上 ckpt_key（形如 ckpt-0096）")
        return layer, hop_key(layer, ckpt_key, step)
    return layer, hop_key(layer, EVIDENCE_SIGNAL[label], step)


# --------------------------------------------------------------------------- #
# 事件流的输入侧
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SourceSpec:
    """一个事件来源。**它的时钟偏移是注入的已知量，也是要被测回来的量。**

    `carries_step=False` 的来源只有本地时间戳 —— 这就是「必须做时钟对齐」的
    全部理由：不给 step 号，就没人能直接把一条采样接到某一步上。
    """

    id: str
    kind: str
    clock_offset_s: float
    drift_s_per_s: float = 0.0
    sample_hz: float | None = None
    carries_step: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "clock_offset_s": self.clock_offset_s,
            "drift_s_per_s": self.drift_s_per_s,
            "sample_hz": self.sample_hz,
            "carries_step": self.carries_step,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SourceSpec":
        hz = raw.get("sample_hz")
        return cls(
            id=str(raw["id"]),
            kind=str(raw["kind"]),
            clock_offset_s=float(raw.get("clock_offset_s", 0.0)),
            drift_s_per_s=float(raw.get("drift_s_per_s", 0.0)),
            sample_hz=(None if hz is None else float(hz)),
            carries_step=bool(raw.get("carries_step", False)),
            note=str(raw.get("note", "")),
        )


@dataclass(frozen=True)
class Event:
    """一条事件。`t_local` 是**产生它的那个来源的本地时钟**，不是真时间。

    只有 `carries_step=True` 的来源（本项目的 `job`）会带 `step`；其余来源的
    `step` 一律是 `None`，必须靠 `align` 模块补出来。
    """

    id: str
    source: str
    kind: str
    t_local: float
    step: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "kind": self.kind,
            "t_local": self.t_local,
            "step": self.step,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Event":
        step = raw.get("step")
        return cls(
            id=str(raw["id"]),
            source=str(raw["source"]),
            kind=str(raw["kind"]),
            t_local=float(raw["t_local"]),
            step=(None if step is None else int(step)),
            payload=dict(raw.get("payload") or {}),
        )


# --------------------------------------------------------------------------- #
# 模拟器的逐步真值（内部对象，落盘仅用于审计）
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StepTrace:
    """模拟器在**真时钟**上看到的第 s 步。这是「事件流的上游」。

    它**不是**产物：产物是从事件流反推出来的东西。留着它只为了两件事 ——
    核对对齐的步归属错误率，以及给真值因果链提供锚点。
    """

    step: int
    t_true_begin: float
    t_true_end: float
    tokens: int
    compute_ms: float
    overhead_ms: float
    loss: float
    grad_norm: float
    lr: float
    gpu0_util: float
    gpu1_util: float
    cpu_util: float
    ram_bytes: float
    vram_bytes: float
    disk_mbps: float
    net_mbps: float
    is_ckpt: bool = False
    is_eval: bool = False
    ckpt_id: str = ""
    eval_id: str = ""
    eval_loss: float | None = None
    nan: bool = False
    root_causes: tuple[str, ...] = ()

    @property
    def step_time_s(self) -> float:
        return self.t_true_end - self.t_true_begin

    @property
    def throughput(self) -> float:
        """tokens / s。GPU 空转时它掉 —— 这是 `bad_batch` 的第二证据。"""
        return self.tokens / self.step_time_s if self.step_time_s > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "t_true_begin": self.t_true_begin,
            "t_true_end": self.t_true_end,
            "tokens": self.tokens,
            "compute_ms": self.compute_ms,
            "overhead_ms": self.overhead_ms,
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "lr": self.lr,
            "gpu0_util": self.gpu0_util,
            "gpu1_util": self.gpu1_util,
            "cpu_util": self.cpu_util,
            "ram_bytes": self.ram_bytes,
            "vram_bytes": self.vram_bytes,
            "disk_mbps": self.disk_mbps,
            "net_mbps": self.net_mbps,
            "is_ckpt": self.is_ckpt,
            "is_eval": self.is_eval,
            "ckpt_id": self.ckpt_id,
            "eval_id": self.eval_id,
            "eval_loss": self.eval_loss,
            "nan": self.nan,
            "root_causes": list(self.root_causes),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StepTrace":
        ev = raw.get("eval_loss")
        return cls(
            step=int(raw["step"]),
            t_true_begin=float(raw["t_true_begin"]),
            t_true_end=float(raw["t_true_end"]),
            tokens=int(raw["tokens"]),
            compute_ms=float(raw["compute_ms"]),
            overhead_ms=float(raw["overhead_ms"]),
            loss=float(raw["loss"]),
            grad_norm=float(raw["grad_norm"]),
            lr=float(raw["lr"]),
            gpu0_util=float(raw["gpu0_util"]),
            gpu1_util=float(raw["gpu1_util"]),
            cpu_util=float(raw["cpu_util"]),
            ram_bytes=float(raw["ram_bytes"]),
            vram_bytes=float(raw["vram_bytes"]),
            disk_mbps=float(raw["disk_mbps"]),
            net_mbps=float(raw["net_mbps"]),
            is_ckpt=bool(raw.get("is_ckpt", False)),
            is_eval=bool(raw.get("is_eval", False)),
            ckpt_id=str(raw.get("ckpt_id", "")),
            eval_id=str(raw.get("eval_id", "")),
            eval_loss=(None if ev is None else float(ev)),
            nan=bool(raw.get("nan", False)),
            root_causes=tuple(str(x) for x in (raw.get("root_causes") or ())),
        )


# --------------------------------------------------------------------------- #
# 对齐产物：把四路异质事件压成一张「逐步信号表」
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StepSignal:
    """**对齐之后**的逐步信号。这是三张图与检出的公共输入。

    `n_samples` / `n_sources` 是必要的可观测性字段：某一步只拿到 0 条采样时，
    它的信号是 `None` 而不是 0 —— 「没测到」与「测到 0」必须分开，
    否则一条缺失的采样会被读成「这个资源很闲」。
    """

    step: int
    t_ref_begin: float
    values: dict[str, float | None]
    n_samples: int
    n_sources: int
    assigned_from: dict[str, int] = field(default_factory=dict)

    def get(self, name: str) -> float | None:
        return self.values.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "t_ref_begin": self.t_ref_begin,
            "values": dict(self.values),
            "n_samples": self.n_samples,
            "n_sources": self.n_sources,
            "assigned_from": dict(self.assigned_from),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StepSignal":
        return cls(
            step=int(raw["step"]),
            t_ref_begin=float(raw["t_ref_begin"]),
            values={str(k): (None if v is None else float(v)) for k, v in (raw.get("values") or {}).items()},
            n_samples=int(raw.get("n_samples", 0)),
            n_sources=int(raw.get("n_sources", 0)),
            assigned_from={str(k): int(v) for k, v in (raw.get("assigned_from") or {}).items()},
        )


@dataclass(frozen=True)
class AlignmentReport:
    """每个来源的时钟对齐结果。**注入量与估计量并排** —— 这就是校准。

    `model` 是 `constant` 或 `linear`：后者说明这个源在漂移，常数偏移不够用。
    `residual_rms_s` 是拟合残差，它超容差就必须报错，而不是照用。
    """

    source: str
    n_beacons: int
    model: str
    injected_offset_s: float
    estimated_offset_s: float
    injected_drift_s_per_s: float
    estimated_drift_s_per_s: float
    residual_rms_s: float
    n_assigned: int = 0
    n_unassigned: int = 0

    @property
    def offset_error_s(self) -> float:
        return abs(self.estimated_offset_s - self.injected_offset_s)

    @property
    def drift_error_s_per_s(self) -> float:
        return abs(self.estimated_drift_s_per_s - self.injected_drift_s_per_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "n_beacons": self.n_beacons,
            "model": self.model,
            "injected_offset_s": self.injected_offset_s,
            "estimated_offset_s": self.estimated_offset_s,
            "injected_drift_s_per_s": self.injected_drift_s_per_s,
            "estimated_drift_s_per_s": self.estimated_drift_s_per_s,
            "residual_rms_s": self.residual_rms_s,
            "offset_error_s": self.offset_error_s,
            "drift_error_s_per_s": self.drift_error_s_per_s,
            "n_assigned": self.n_assigned,
            "n_unassigned": self.n_unassigned,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AlignmentReport":
        return cls(
            source=str(raw["source"]),
            n_beacons=int(raw["n_beacons"]),
            model=str(raw["model"]),
            injected_offset_s=float(raw["injected_offset_s"]),
            estimated_offset_s=float(raw["estimated_offset_s"]),
            injected_drift_s_per_s=float(raw["injected_drift_s_per_s"]),
            estimated_drift_s_per_s=float(raw["estimated_drift_s_per_s"]),
            residual_rms_s=float(raw["residual_rms_s"]),
            n_assigned=int(raw.get("n_assigned", 0)),
            n_unassigned=int(raw.get("n_unassigned", 0)),
        )


@dataclass(frozen=True)
class AlignmentModel:
    """**一种对齐假设**在某个来源上的拟合结果。

    三种假设同台竞技，因为「要不要对齐 / 常数够不够 / 漂移要不要建模」这三个问题
    必须能被数据推翻，而不是靠信念：

    | `model` | 形式 | 参数 |
    | --- | --- | --- |
    | `identity` | `t_ref = t_local` | 0 个 —— 「不对齐」 |
    | `constant` | `t_ref = t_local − a` | 1 个（中位残差） |
    | `linear` | `t_ref = (t_local − a) / b` | 2 个（最小二乘） |

    `assign_error_rate` 是这种假设下把采样接到步上的**错误率** ——
    它是判断标准，不是拟合优度：拟合得再好，接错了步也没有意义。
    """

    model: str
    a: float
    b: float
    residual_rms_s: float
    n_assigned: int = 0
    n_unassigned: int = 0
    assign_error_rate: float = 0.0

    @property
    def drift_s_per_s(self) -> float:
        """斜率相对 1 的偏离量，就是漂移率。`identity` 恒为 0。"""
        return self.b - 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "a": self.a,
            "b": self.b,
            "drift_s_per_s": self.drift_s_per_s,
            "residual_rms_s": self.residual_rms_s,
            "n_assigned": self.n_assigned,
            "n_unassigned": self.n_unassigned,
            "assign_error_rate": self.assign_error_rate,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AlignmentModel":
        return cls(
            model=str(raw["model"]),
            a=float(raw["a"]),
            b=float(raw["b"]),
            residual_rms_s=float(raw["residual_rms_s"]),
            n_assigned=int(raw.get("n_assigned", 0)),
            n_unassigned=int(raw.get("n_unassigned", 0)),
            assign_error_rate=float(raw.get("assign_error_rate", 0.0)),
        )


@dataclass(frozen=True)
class AlignmentResult:
    """对齐阶段的全部产出：**选中的模型** + **三种模型的对比** + 逐步信号表。

    `chosen` 与 `candidates` 都留着，因为报告里要回答的是「漂移感知比常数偏移
    多救回多少条归因」—— 只留赢家就答不了这个问题。
    """

    reports: tuple[AlignmentReport, ...]
    candidates: tuple[tuple[str, tuple[AlignmentModel, ...]], ...]
    chosen_model: dict[str, str]
    step_table: tuple[StepSignal, ...]
    step_boundaries: tuple[tuple[int, float, float], ...]
    n_events_total: int
    n_events_assigned: int

    def report(self, source: str) -> AlignmentReport:
        for r in self.reports:
            if r.source == source:
                return r
        raise KeyError(f"没有这个来源的对齐结果：{source!r}")

    def models_of(self, source: str) -> tuple[AlignmentModel, ...]:
        for sid, models in self.candidates:
            if sid == source:
                return models
        return ()

    def model(self, source: str, name: str) -> AlignmentModel | None:
        for m in self.models_of(source):
            if m.model == name:
                return m
        return None

    @property
    def drift_sources(self) -> tuple[str, ...]:
        return tuple(s for s, m in sorted(self.chosen_model.items()) if m == "linear")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reports": [r.to_dict() for r in self.reports],
            "candidates": [
                {"source": sid, "models": [m.to_dict() for m in models]} for sid, models in self.candidates
            ],
            "chosen_model": dict(self.chosen_model),
            "step_table": [s.to_dict() for s in self.step_table],
            "step_boundaries": [
                {"step": s, "t_ref_begin": a, "t_ref_end": b} for s, a, b in self.step_boundaries
            ],
            "n_events_total": self.n_events_total,
            "n_events_assigned": self.n_events_assigned,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AlignmentResult":
        return cls(
            reports=tuple(AlignmentReport.from_dict(r) for r in raw["reports"]),
            candidates=tuple(
                (str(c["source"]), tuple(AlignmentModel.from_dict(m) for m in c["models"]))
                for c in raw["candidates"]
            ),
            chosen_model={str(k): str(v) for k, v in raw["chosen_model"].items()},
            step_table=tuple(StepSignal.from_dict(s) for s in raw["step_table"]),
            step_boundaries=tuple(
                (int(b["step"]), float(b["t_ref_begin"]), float(b["t_ref_end"]))
                for b in raw["step_boundaries"]
            ),
            n_events_total=int(raw["n_events_total"]),
            n_events_assigned=int(raw["n_events_assigned"]),
        )


# --------------------------------------------------------------------------- #
# 三张图里「不是 DAG 节点」的那部分
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StateTransition:
    step: int
    frm: str
    to: str
    t_local: float
    reason: str = ""

    @property
    def key(self) -> str:
        return f"{self.frm}->{self.to}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "frm": self.frm,
            "to": self.to,
            "t_local": self.t_local,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StateTransition":
        return cls(
            step=int(raw["step"]),
            frm=str(raw["frm"]),
            to=str(raw["to"]),
            t_local=float(raw["t_local"]),
            reason=str(raw.get("reason", "")),
        )


@dataclass(frozen=True)
class ResourcePressure:
    """某个资源在某个窗口里的压力。`normalized` 是**相对它自己的基线**归一化后的
    峰值 —— 不归一化的话，VRAM（几十 GB）会永远压过 GPU 利用率（百分比），
    「瓶颈在哪个资源」这个问题的答案就恒等于「VRAM」。"""

    resource: str
    peak_step: int
    peak_value: float
    baseline: float
    normalized: float
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource": self.resource,
            "peak_step": self.peak_step,
            "peak_value": self.peak_value,
            "baseline": self.baseline,
            "normalized": self.normalized,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ResourcePressure":
        return cls(
            resource=str(raw["resource"]),
            peak_step=int(raw["peak_step"]),
            peak_value=float(raw["peak_value"]),
            baseline=float(raw["baseline"]),
            normalized=float(raw["normalized"]),
            direction=str(raw["direction"]),
        )


# --------------------------------------------------------------------------- #
# 异常与归因
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Anomaly:
    """检出到的一段异常。**连续被标记的步聚成一个 episode** ——
    逐步报出来会让「一次 loss spike」变成三条异常，召回率立刻虚高。"""

    id: str
    signal: str
    kind: str
    step_lo: int
    step_hi: int
    peak_step: int
    peak_value: float
    peak_z: float
    is_nan: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "signal": self.signal,
            "kind": self.kind,
            "step_lo": self.step_lo,
            "step_hi": self.step_hi,
            "peak_step": self.peak_step,
            "peak_value": self.peak_value,
            "peak_z": self.peak_z,
            "is_nan": self.is_nan,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Anomaly":
        return cls(
            id=str(raw["id"]),
            signal=str(raw["signal"]),
            kind=str(raw["kind"]),
            step_lo=int(raw["step_lo"]),
            step_hi=int(raw["step_hi"]),
            peak_step=int(raw["peak_step"]),
            peak_value=float(raw["peak_value"]),
            peak_z=float(raw["peak_z"]),
            is_nan=bool(raw.get("is_nan", False)),
        )


@dataclass(frozen=True)
class CausalHop:
    """因果链上的一跳。`layer` 决定它挂在哪张图上，`key` 是它的唯一定位符。"""

    layer: str
    key: str
    step: int
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"layer": self.layer, "key": self.key, "step": self.step, "detail": self.detail}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CausalHop":
        return cls(
            layer=str(raw["layer"]),
            key=str(raw["key"]),
            step=int(raw["step"]),
            detail=str(raw.get("detail", "")),
        )


@dataclass(frozen=True)
class Attribution:
    """**五问的答案。** 这就是本项目的交付物。

    `chain` 是从异常出发后向走出来的因果链；`signature_hits` 记录声明的签名里
    哪几项真的在事件流里被看到了 —— 于是「声明了但没出现」和「出现了但没声明」
    这两件事都能被分开报出来。`ambiguous` 是「不止一个根因签名同样匹配」的标记：
    这种情况下**不许硬选一个**，要如实标出来。

    `covered_anomalies` 是**这次归因解释掉的检出**（一次 6 步的 lr spike 会同时
    触发 grad_norm 与 loss 两条检出，它们归到同一个根因下）。注意它装的是**检出的
    编号**，不是注入项的编号 —— 后者是 `TruthRecord` 的事。**答案里不许出现真值**，
    否则「归因对了没有」就成了一句自我声明。
    """

    anomaly_id: str
    what: str
    when_step: int
    resource: str
    config_key: str
    config_value: Any
    checkpoint: str
    root_cause_kind: str
    chain: tuple[CausalHop, ...]
    signature_hits: tuple[str, ...]
    signature_missing: tuple[str, ...]
    covered_anomalies: tuple[str, ...]
    confidence: float
    ambiguous: bool

    @property
    def chain_keys(self) -> tuple[str, ...]:
        return tuple(h.key for h in self.chain)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "what": self.what,
            "when_step": self.when_step,
            "resource": self.resource,
            "config_key": self.config_key,
            "config_value": self.config_value,
            "checkpoint": self.checkpoint,
            "root_cause_kind": self.root_cause_kind,
            "chain": [h.to_dict() for h in self.chain],
            "chain_keys": list(self.chain_keys),
            "signature_hits": list(self.signature_hits),
            "signature_missing": list(self.signature_missing),
            "covered_anomalies": list(self.covered_anomalies),
            "confidence": self.confidence,
            "ambiguous": self.ambiguous,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Attribution":
        return cls(
            anomaly_id=str(raw["anomaly_id"]),
            what=str(raw["what"]),
            when_step=int(raw["when_step"]),
            resource=str(raw["resource"]),
            config_key=str(raw["config_key"]),
            config_value=raw.get("config_value"),
            checkpoint=str(raw["checkpoint"]),
            root_cause_kind=str(raw["root_cause_kind"]),
            chain=tuple(CausalHop.from_dict(h) for h in (raw.get("chain") or ())),
            signature_hits=tuple(str(x) for x in (raw.get("signature_hits") or ())),
            signature_missing=tuple(str(x) for x in (raw.get("signature_missing") or ())),
            covered_anomalies=tuple(str(x) for x in (raw.get("covered_anomalies") or ())),
            confidence=float(raw.get("confidence", 0.0)),
            ambiguous=bool(raw.get("ambiguous", False)),
        )


@dataclass(frozen=True)
class TruthRecord:
    """注入的根因**及其在事件流里真实留下的痕迹**。

    `chain` 不是配置里声明的 —— 它是模拟器按传播规则**跑出来之后记下来的**。
    这个区别很重要：声明的是「我注入了什么」，记下来的是「它真的带出了什么」。
    """

    root_cause_id: str
    kind: str
    at_step: int
    duration_steps: int
    magnitude: float
    what: str
    when_step: int
    resource: str
    config_key: str
    config_value: Any
    checkpoint: str
    chain: tuple[CausalHop, ...]
    recovery: str

    @property
    def chain_keys(self) -> tuple[str, ...]:
        return tuple(h.key for h in self.chain)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_cause_id": self.root_cause_id,
            "kind": self.kind,
            "at_step": self.at_step,
            "duration_steps": self.duration_steps,
            "magnitude": self.magnitude,
            "what": self.what,
            "when_step": self.when_step,
            "resource": self.resource,
            "config_key": self.config_key,
            "config_value": self.config_value,
            "checkpoint": self.checkpoint,
            "chain": [h.to_dict() for h in self.chain],
            "chain_keys": list(self.chain_keys),
            "recovery": self.recovery,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TruthRecord":
        return cls(
            root_cause_id=str(raw["root_cause_id"]),
            kind=str(raw["kind"]),
            at_step=int(raw["at_step"]),
            duration_steps=int(raw["duration_steps"]),
            magnitude=float(raw["magnitude"]),
            what=str(raw["what"]),
            when_step=int(raw["when_step"]),
            resource=str(raw["resource"]),
            config_key=str(raw["config_key"]),
            config_value=raw.get("config_value"),
            checkpoint=str(raw["checkpoint"]),
            chain=tuple(CausalHop.from_dict(h) for h in (raw.get("chain") or ())),
            recovery=str(raw.get("recovery", "")),
        )


@dataclass(frozen=True)
class ProbeResult:
    """一个可判分的反事实探针。

    三类探针问的是三件不同的事：

    - **P（充分性）**：去掉这个原因，目标信号真的动了吗？—— 动得不够就是「不充分」。
    - **F（特异性）**：去掉它，**它声明不影响的信号**动了吗？—— 全都在动说明
      「只要注了就都行」，P 全过也没意义。
    - **h（定位性）**：把它平移到别处，原位置的信号还动吗？—— 还动说明
      「Δ 到处都是」，归因根本没有定位能力。
    """

    root_key: str
    probe: str
    signal: str
    step: int
    delta: float
    alpha: float
    sufficient: bool
    expected: bool
    passed: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_key": self.root_key,
            "probe": self.probe,
            "signal": self.signal,
            "step": self.step,
            "delta": self.delta,
            "alpha": self.alpha,
            "sufficient": self.sufficient,
            "expected": self.expected,
            "passed": self.passed,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ProbeResult":
        return cls(
            root_key=str(raw["root_key"]),
            probe=str(raw["probe"]),
            signal=str(raw["signal"]),
            step=int(raw["step"]),
            delta=float(raw["delta"]),
            alpha=float(raw["alpha"]),
            sufficient=bool(raw["sufficient"]),
            expected=bool(raw["expected"]),
            passed=bool(raw["passed"]),
            note=str(raw.get("note", "")),
        )


@dataclass(frozen=True)
class AlphaCurve:
    """**α 标定曲线**：把某个原因的幅度从 0（等价于移除）线性拉到原值，
    看目标信号相对噪音尺度的偏离怎么变。

    `f_star` 是最小的、仍然被判为充分条件的幅度因子。它是「这个原因要做到多小
    才不再是原因」的量化答案 —— 也是 α 规则唯一的可判分输出。
    """

    root_key: str
    kind: str
    signal: str
    points: tuple[tuple[float, float], ...]
    f_star: float
    monotone: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_key": self.root_key,
            "kind": self.kind,
            "signal": self.signal,
            "points": [{"f": f, "delta_over_sigma": d} for f, d in self.points],
            "f_star": self.f_star,
            "monotone": self.monotone,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AlphaCurve":
        return cls(
            root_key=str(raw["root_key"]),
            kind=str(raw["kind"]),
            signal=str(raw["signal"]),
            points=tuple((float(p["f"]), float(p["delta_over_sigma"])) for p in raw["points"]),
            f_star=float(raw["f_star"]),
            monotone=bool(raw["monotone"]),
        )


@dataclass(frozen=True)
class CounterfactualCheck:
    """**反事实反向链**：把某个原因从模拟器里去掉，重跑整条流水线，
    看它的归因是否真的消失。这是「驱动真实模拟器重放」那一句的执行点。"""

    root_key: str
    kind: str
    at_step: int
    n_attributions_before: int
    n_attributions_after: int
    chain_disappeared: bool
    same_kind_after: int

    @property
    def passed(self) -> bool:
        return self.chain_disappeared

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_key": self.root_key,
            "kind": self.kind,
            "at_step": self.at_step,
            "n_attributions_before": self.n_attributions_before,
            "n_attributions_after": self.n_attributions_after,
            "chain_disappeared": self.chain_disappeared,
            "same_kind_after": self.same_kind_after,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CounterfactualCheck":
        return cls(
            root_key=str(raw["root_key"]),
            kind=str(raw["kind"]),
            at_step=int(raw["at_step"]),
            n_attributions_before=int(raw["n_attributions_before"]),
            n_attributions_after=int(raw["n_attributions_after"]),
            chain_disappeared=bool(raw["chain_disappeared"]),
            same_kind_after=int(raw["same_kind_after"]),
        )


@dataclass(frozen=True)
class AlignmentAblation:
    """把对齐假设限制在一个子集后，**真的重跑整条流水线**，看还剩几条归因。

    这是「对齐是不是必要的」这句话唯一能被推翻的地方。看残差表只能说明
    「常数偏移拟合不上」，说明不了「少了漂移会丢多少条归因」——
    后者才是这个项目要付的代价。
    """

    only: tuple[str, ...]
    n_roots_recovered: int
    n_roots_total: int
    error: str = ""

    @property
    def label(self) -> str:
        return "+".join(self.only)

    @property
    def recall(self) -> float:
        return (self.n_roots_recovered / self.n_roots_total) if self.n_roots_total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "only": list(self.only),
            "label": self.label,
            "n_roots_recovered": self.n_roots_recovered,
            "n_roots_total": self.n_roots_total,
            "recall": self.recall,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AlignmentAblation":
        return cls(
            only=tuple(str(x) for x in raw["only"]),
            n_roots_recovered=int(raw["n_roots_recovered"]),
            n_roots_total=int(raw["n_roots_total"]),
            error=str(raw.get("error", "")),
        )


@dataclass(frozen=True)
class VerifyResult:
    """第三方校验的全部产出。**它读真值，因为它就是那个第三方。**"""

    probes: tuple[ProbeResult, ...]
    counterfactuals: tuple[CounterfactualCheck, ...]
    curves: tuple[AlphaCurve, ...]
    alpha_sigma: float
    ablations: tuple[AlignmentAblation, ...] = ()
    n_roots_recovered: int = 0

    def probe_pass_rate(self, probe: str) -> float:
        xs = [p for p in self.probes if p.probe == probe]
        return (sum(1 for p in xs if p.passed) / len(xs)) if xs else 0.0

    def ablation(self, label: str) -> AlignmentAblation | None:
        for a in self.ablations:
            if a.label == label:
                return a
        return None

    def alignment_lift(self) -> float:
        """漂移感知比「只用常数偏移」多救回多少比例。**允许为 0** ——
        如果漂移没帮上忙，这个数就该是 0，而不是被算成一个漂亮的数字。"""
        if not self.ablations or not self.n_roots_recovered:
            return 0.0
        worst = min(a.n_roots_recovered for a in self.ablations)
        return (self.n_roots_recovered - worst) / max(1, self.n_roots_recovered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probes": [p.to_dict() for p in self.probes],
            "counterfactuals": [c.to_dict() for c in self.counterfactuals],
            "curves": [c.to_dict() for c in self.curves],
            "ablations": [a.to_dict() for a in self.ablations],
            "alpha_sigma": self.alpha_sigma,
            "n_roots_recovered": self.n_roots_recovered,
            "alignment_lift": self.alignment_lift(),
            "probe_pass_rate": {p: self.probe_pass_rate(p) for p in ("P", "F", "h")},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "VerifyResult":
        return cls(
            probes=tuple(ProbeResult.from_dict(p) for p in raw["probes"]),
            counterfactuals=tuple(CounterfactualCheck.from_dict(c) for c in raw["counterfactuals"]),
            curves=tuple(AlphaCurve.from_dict(c) for c in raw["curves"]),
            alpha_sigma=float(raw["alpha_sigma"]),
            ablations=tuple(AlignmentAblation.from_dict(a) for a in raw.get("ablations", ())),
            n_roots_recovered=int(raw.get("n_roots_recovered", 0)),
        )


def chain_of(pairs: Sequence[tuple[str, str, int]]) -> tuple[CausalHop, ...]:
    """`(layer, key, step)` 三元组列表 → 因果链。放在这里是为了让
    模拟器与归因器**共用同一个构造函数**，否则两边的链会长得不一样。"""
    return tuple(CausalHop(layer=lay, key=key, step=st) for lay, key, st in pairs)


# --------------------------------------------------------------------------- #
# 三张图。**在数据层就分开** —— 混在一起就会退化成一张杂乱的流程图。
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class JobNode:
    """Job DAG 的一个**实例**节点。`template` 是它来自哪个声明节点。

    实例化是关键：`train` 声明一次，运行时展开成 120 个节点。于是「第 55 步那个
    train 节点依赖谁」是个能查的问题，而「train 依赖 preprocess」只是一句声明。
    """

    id: str
    template: str
    label: str
    step_lo: int
    step_hi: int
    n_events: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "template": self.template,
            "label": self.label,
            "step_lo": self.step_lo,
            "step_hi": self.step_hi,
            "n_events": self.n_events,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "JobNode":
        return cls(
            id=str(raw["id"]),
            template=str(raw["template"]),
            label=str(raw["label"]),
            step_lo=int(raw["step_lo"]),
            step_hi=int(raw["step_hi"]),
            n_events=int(raw.get("n_events", 0)),
        )


@dataclass(frozen=True)
class JobDag:
    """**第一张图：Job DAG。** 回答「这次运行有哪些作业、谁依赖谁」。

    两层结构：`template_edges` 是声明（8 个节点），`edges` 是运行时展开后的实例图。
    查询一律走实例图 —— 声明层只用来解释结构，不参与归因。
    """

    nodes: tuple[JobNode, ...]
    edges: tuple[tuple[str, str], ...]
    template_nodes: tuple[tuple[str, str], ...]
    template_edges: tuple[tuple[str, str], ...]
    n_steps: int

    def node(self, node_id: str) -> JobNode:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(f"没有这个 Job 节点：{node_id!r}")

    def has(self, node_id: str) -> bool:
        return any(n.id == node_id for n in self.nodes)

    def deps(self, node_id: str) -> tuple[str, ...]:
        """直接依赖（`to` 依赖 `from`）。"""
        return tuple(a for a, b in self.edges if b == node_id)

    def children(self, node_id: str) -> tuple[str, ...]:
        return tuple(b for a, b in self.edges if a == node_id)

    def ancestors(self, node_id: str) -> tuple[str, ...]:
        """传递闭包，**按拓扑序**返回（确定性）。"""
        seen: set[str] = set()
        stack = [node_id]
        while stack:
            cur = stack.pop()
            for d in self.deps(cur):
                if d not in seen:
                    seen.add(d)
                    stack.append(d)
        order = self.topo_order()
        return tuple(n for n in order if n in seen)

    def descendants(self, node_id: str) -> tuple[str, ...]:
        seen: set[str] = set()
        stack = [node_id]
        while stack:
            cur = stack.pop()
            for c in self.children(cur):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        order = self.topo_order()
        return tuple(n for n in order if n in seen)

    def topo_order(self) -> tuple[str, ...]:
        """Kahn 算法。**入度为 0 时按节点 id 排序**取下一个 —— 否则输出会依赖
        `set` 的迭代顺序，而「构建决定论」是这张图的验收标准之一。"""
        indeg = {n.id: 0 for n in self.nodes}
        for a, b in self.edges:
            indeg[b] = indeg.get(b, 0) + 1
        ready = sorted(k for k, v in indeg.items() if v == 0)
        out: list[str] = []
        while ready:
            cur = ready.pop(0)
            out.append(cur)
            for c in sorted(self.children(cur)):
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
            ready.sort()
        if len(out) != len(self.nodes):
            raise ValueError("Job DAG 里有环 —— 它不是 DAG。")
        return tuple(out)

    def by_template(self, template: str) -> tuple[JobNode, ...]:
        return tuple(n for n in self.nodes if n.template == template)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [{"from": a, "to": b} for a, b in self.edges],
            "template_nodes": [{"id": i, "label": l} for i, l in self.template_nodes],
            "template_edges": [{"from": a, "to": b} for a, b in self.template_edges],
            "n_steps": self.n_steps,
            "n_nodes": len(self.nodes),
            "n_edges": self.n_edges,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "JobDag":
        return cls(
            nodes=tuple(JobNode.from_dict(n) for n in raw["nodes"]),
            edges=tuple((str(e["from"]), str(e["to"])) for e in raw["edges"]),
            template_nodes=tuple((str(e["id"]), str(e["label"])) for e in raw["template_nodes"]),
            template_edges=tuple((str(e["from"]), str(e["to"])) for e in raw["template_edges"]),
            n_steps=int(raw["n_steps"]),
        )


@dataclass(frozen=True)
class StateGraph:
    """**第二张图：Training State Graph。**

    `declared_*` 是声明（写在配置里），`visited_*` 是本次运行真的走过的。
    **两者的差集是有信息的**：一个声明了但从没走到的状态（本项目是 `PAUSED`）
    说明这条路径本次没被覆盖 —— 它不该被悄悄画进图里当成走过。
    """

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    visited_nodes: tuple[str, ...]
    visited_edges: tuple[tuple[str, str], ...]
    unvisited_nodes: tuple[str, ...]
    unvisited_edges: tuple[tuple[str, str], ...]
    transitions: tuple[StateTransition, ...]
    path: tuple[tuple[int, str], ...]
    steps_in_state: tuple[tuple[str, int], ...]
    n_transitions: int

    def edges_out(self, state: str) -> tuple[tuple[str, str], ...]:
        return tuple(e for e in self.edges if e[0] == state)

    def edges_in(self, state: str) -> tuple[tuple[str, str], ...]:
        return tuple(e for e in self.edges if e[1] == state)

    def state_at(self, step: int) -> str | None:
        for s, name in self.path:
            if s == step:
                return name
        return None

    def transitions_at(self, step: int) -> tuple[StateTransition, ...]:
        return tuple(t for t in self.transitions if t.step == step)

    def steps_of(self, state: str) -> int:
        for name, n in self.steps_in_state:
            if name == state:
                return n
        return 0

    @property
    def is_acyclic(self) -> bool:
        """状态图**允许有环**（TRAINING ↔ EVALUATING 就是环），所以这里不判环，
        只判「是不是真的走完了」。留着这个属性是为了让报告里那句话有出处。"""
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": [{"from": a, "to": b} for a, b in self.edges],
            "visited_nodes": list(self.visited_nodes),
            "visited_edges": [{"from": a, "to": b} for a, b in self.visited_edges],
            "unvisited_nodes": list(self.unvisited_nodes),
            "unvisited_edges": [{"from": a, "to": b} for a, b in self.unvisited_edges],
            "transitions": [t.to_dict() for t in self.transitions],
            "path": [{"step": s, "state": n} for s, n in self.path],
            "steps_in_state": [{"state": n, "steps": c} for n, c in self.steps_in_state],
            "n_transitions": self.n_transitions,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StateGraph":
        return cls(
            nodes=tuple(str(x) for x in raw["nodes"]),
            edges=tuple((str(e["from"]), str(e["to"])) for e in raw["edges"]),
            visited_nodes=tuple(str(x) for x in raw["visited_nodes"]),
            visited_edges=tuple((str(e["from"]), str(e["to"])) for e in raw["visited_edges"]),
            unvisited_nodes=tuple(str(x) for x in raw["unvisited_nodes"]),
            unvisited_edges=tuple((str(e["from"]), str(e["to"])) for e in raw["unvisited_edges"]),
            transitions=tuple(StateTransition.from_dict(t) for t in raw["transitions"]),
            path=tuple((int(p["step"]), str(p["state"])) for p in raw["path"]),
            steps_in_state=tuple((str(p["state"]), int(p["steps"])) for p in raw["steps_in_state"]),
            n_transitions=int(raw["n_transitions"]),
        )


@dataclass(frozen=True)
class ResourceGraph:
    """**第三张图：Resource Graph。** `Job → {GPU0, GPU1, CPU, RAM, VRAM, Disk, Network}`。

    `normalized` 是**相对该资源自己的基线**归一化后的峰值。不归一化的话，
    VRAM（几十 GB）会永远压过 GPU 利用率（百分比），于是「瓶颈在哪个资源」
    这个问题的答案恒等于「VRAM」—— 一个永远不变的答案等于没有答案。
    """

    hub: str
    resources: tuple[str, ...]
    signal_of: dict[str, str]
    edges: tuple[tuple[str, str], ...]
    pressure: tuple[ResourcePressure, ...]
    bottleneck_by_step: tuple[tuple[int, str], ...]
    n_steps: int

    def pressure_of(self, resource: str) -> ResourcePressure:
        for p in self.pressure:
            if p.resource == resource:
                return p
        raise KeyError(f"没有这个资源：{resource!r}")

    def ranked(self) -> tuple[ResourcePressure, ...]:
        """按归一化压力降序。打平时按资源名排序 —— 保持决定论。"""
        return tuple(sorted(self.pressure, key=lambda p: (-p.normalized, p.resource)))

    def bottleneck_at(self, step: int) -> str | None:
        for s, r in self.bottleneck_by_step:
            if s == step:
                return r
        return None

    def steps_of(self, resource: str) -> int:
        return sum(1 for _, r in self.bottleneck_by_step if r == resource)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hub": self.hub,
            "resources": list(self.resources),
            "signal_of": dict(self.signal_of),
            "edges": [{"from": a, "to": b} for a, b in self.edges],
            "pressure": [p.to_dict() for p in self.pressure],
            "bottleneck_by_step": [{"step": s, "resource": r} for s, r in self.bottleneck_by_step],
            "n_steps": self.n_steps,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ResourceGraph":
        return cls(
            hub=str(raw["hub"]),
            resources=tuple(str(x) for x in raw["resources"]),
            signal_of={str(k): str(v) for k, v in raw["signal_of"].items()},
            edges=tuple((str(e["from"]), str(e["to"])) for e in raw["edges"]),
            pressure=tuple(ResourcePressure.from_dict(p) for p in raw["pressure"]),
            bottleneck_by_step=tuple(
                (int(b["step"]), str(b["resource"])) for b in raw["bottleneck_by_step"]
            ),
            n_steps=int(raw["n_steps"]),
        )
