"""配置：口径、门槛、注入量。

**所有阈值都在这里，代码里不硬编码。** 本项目尤其如此 —— 因为「注入量」本身就是
配置：时钟偏移、漂移率、根因的位置与幅度全写在 `configs/default.json` 里，
流水线的任务是把它们**测回来**。注入量一旦进了代码，「校准」就变成了自我声明。

配置里有一条刻意的分界：

- **`causal_model`** 是**通用知识** —— 每一「种」根因会牵连哪个配置键、哪个资源、
  会留下什么有序签名。复原器只看得见它。
- **`root_causes`** 是**这一次的注入实例** —— 哪个种类、在第几步、多大、持续多久。

把两者混成一块（把 `config_key` / `resource` 直接写进注入项）会让复原器**抄答案**：
它不再是「用因果模型去解释事件流」，而是「把答案从配置里读出来」。这一条是本项目
唯一一处「数据结构决定了结论有没有意义」的地方。

`protocol_digest` 只吃**口径**（detect / align / causal / causal_model / acceptance），
`version_digest` 吃**这次跑的全部输入**（run / sources / beacon / root_causes + 种子）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .rng import canonical_json, join_digest, sha256_text
from .schema import EVIDENCE_LAYER, RESOURCES, ROOT_CAUSE_KINDS, SIGNALS, STATES

__all__ = [
    "DEFAULTS",
    "SOURCE_KINDS",
    "deep_merge",
    "load_config",
    "validate_config",
    "resolve_path",
    "protocol_digest",
    "version_digest",
    "sources_of",
    "root_causes_of",
    "causal_model_of",
    "signatures_of",
    "expectations_of",
]

# 允许的事件来源种类。`job` 必须有且只有一个 —— 它是参考时钟。
SOURCE_KINDS: tuple[str, ...] = ("job", "metric", "resource", "io")

DEFAULTS: dict[str, Any] = {
    "seed": 20260930,
    "run": {
        "steps": 120,
        "batch_size": 8,
        "seq_len": 64,
        "length_cv": 0.32,
        "base_lr": 0.0003,
        "warmup_steps": 8,
        "grad_clip": 1.0,
        "micro_batch": 2,
        "grad_accum": 4,
        "max_seq_len": 256,
        "precision": "bf16",
        "ckpt_every": 12,
        "eval_every": 10,
        "loss0": 4.2,
        "loss_decay": 0.971,
        "loss_noise": 0.045,
        "grad_norm_base": 0.85,
        "grad_norm_sigma": 0.28,
        "base_compute_ms": 300.0,
        "per_token_ms": 0.30,
        "overhead_ms": 120.0,
        "ckpt_write_ms": 520.0,
        "eval_ms": 320.0,
        "recover_ms": 1400.0,
        "bytes_per_param": 2,
        "params_m": 740,
        "vram_opt_mult": 2.0,
        "bytes_per_token_activation": 24000.0,
        "net_mbps_base": 180.0,
        "disk_mbps_write": 420.0,
        "disk_mbps_stall": 24.0,
    },
    "sources": [
        {
            "id": "job",
            "kind": "job",
            "clock_offset_s": 0.0,
            "drift_s_per_s": 0.0,
            "sample_hz": None,
            "carries_step": True,
            "note": "参考时钟：只有它带 step 号，所以偏移不可辨识，按定义取 0。",
        },
        {
            "id": "metric",
            "kind": "metric",
            "clock_offset_s": 0.37,
            "drift_s_per_s": 0.0,
            "sample_hz": 4.0,
            "carries_step": False,
            "note": "loss / grad_norm 采样器。",
        },
        {
            "id": "resource",
            "kind": "resource",
            "clock_offset_s": -0.21,
            "drift_s_per_s": 0.006,
            "sample_hz": 8.0,
            "carries_step": False,
            "note": "GPU/RAM/Disk 采样器。**它在线性漂移** —— 常数偏移不够用。",
        },
        {
            "id": "io",
            "kind": "io",
            "clock_offset_s": 1.13,
            "drift_s_per_s": 0.0,
            "sample_hz": None,
            "carries_step": False,
            "note": "dataloader 读批 / checkpoint 写盘。",
        },
    ],
    "beacon": {
        "every_steps": 12,
        "jitter_s": 0.004,
        "drop_prob": 0.0,
        "note": "同步信标：每个 sink 用自己的本地时钟记一条 sync 记录。序号对应关系不依赖时间戳。",
    },
    "causal_model": {
        "lr_spike": {
            "primary_signal": "loss",
            "config_key": "train.lr",
            "resource": "GPU0",
            "recovery": "revert_config",
            "signature": ["config_change", "grad_norm_z", "loss_z"],
        },
        "bad_batch": {
            "primary_signal": "step_time_s",
            "config_key": "data.max_seq_len",
            "resource": "CPU",
            "recovery": "skip_batch",
            "signature": ["batch_tokens_z", "step_time_z", "cpu_z"],
        },
        "vram_pressure": {
            "primary_signal": "vram_bytes",
            "config_key": "train.micro_batch",
            "resource": "VRAM",
            "recovery": "reduce_micro_batch",
            "signature": ["vram_z", "step_time_z", "gpu_util_z_drop"],
        },
        "disk_stall": {
            "primary_signal": "disk_mbps",
            "config_key": "ckpt.every",
            "resource": "Disk",
            "recovery": "retry_write",
            "signature": ["disk_z_drop", "ckpt_write_z", "step_time_z"],
        },
        "eval_regression": {
            "primary_signal": "eval_loss",
            "config_key": "data.eval_slice",
            "resource": "Disk",
            "recovery": "rerun_eval",
            "signature": ["eval_loss_z", "disk_z_drop", "step_time_z"],
            "note": "第三条签名是**一个人为的信念**：「评测读数变差，评测这一趟也会更慢」。流水线会告诉你它从来没有兑现过。",
        },
        "nan_grad": {
            "primary_signal": "loss",
            "config_key": "train.grad_clip",
            "resource": "GPU0",
            "recovery": "resume_from_checkpoint",
            "signature": ["grad_norm_z", "loss_nan", "state_failed", "gpu_util_z_drop", "ckpt_restore"],
        },
    },
    "root_causes": [
        {
            "id": "rc-lr",
            "kind": "lr_spike",
            "at_step": 42,
            "duration_steps": 6,
            "magnitude": 8.0,
            "note": "学习率 ×8 六步。先炸 grad_norm，再抬 loss。",
        },
        {
            "id": "rc-batch",
            "kind": "bad_batch",
            "at_step": 55,
            "duration_steps": 1,
            "magnitude": 8.0,
            "note": "一个超长批：tokens ×8、host 侧开销 ×10 ⇒ GPU 空转。",
        },
        {
            "id": "rc-vram",
            "kind": "vram_pressure",
            "at_step": 62,
            "duration_steps": 8,
            "magnitude": 1.5,
            "note": "显存压力：VRAM 抬升、分配器重试 ⇒ step_time 变长。与 rc-disk 相隔 2 步，正好检验「事件簇能不能分开」。",
        },
        {
            "id": "rc-disk",
            "kind": "disk_stall",
            "at_step": 72,
            "duration_steps": 3,
            "magnitude": 5.0,
            "note": "正好压在 checkpoint 步上：写盘 ×5，磁盘吞吐塌陷。",
        },
        {
            "id": "rc-eval",
            "kind": "eval_regression",
            "at_step": 90,
            "duration_steps": 1,
            "magnitude": 1.9,
            "note": "评测集读数跳变，只在 eval 步可见。",
        },
        {
            "id": "rc-nan",
            "kind": "nan_grad",
            "at_step": 97,
            "duration_steps": 1,
            "magnitude": 1.0,
            "note": "loss 变 nan ⇒ 状态机走 TRAINING→FAILED→RECOVER→TRAINING。",
        },
    ],
    "detect": {
        "k": 6.0,
        "mad_floor_ratio": 0.02,
        "min_episode_steps": 1,
        "stratify_by_step_type": True,
        "min_stratum_size": 4,
        "detrend": True,
        "detrend_span": 4.0,
        "stratum_fallback": "none",
        "monitored": [
            "loss",
            "grad_norm",
            "step_time_s",
            "gpu0_util",
            "vram_bytes",
            "disk_mbps",
            "ckpt_write_s",
            "eval_loss",
            "batch_tokens",
            "cpu_util",
        ],
        "note": "稳健 z = (x − median) / max(1.4826·MAD, floor)。NaN 一律显式判异常，不走 z。非计算步（ckpt/eval）必须单独建模基线。",
    },
    "align": {
        "residual_tol_s": 0.05,
        "drift_tol_s_per_s": 0.0005,
        "min_beacons": 4,
        "max_step_assign_error": 0.15,
        "note": "残差超容差 / 缺 beacon / 漂移未被建模，都必须报错，不许静默退回 0 偏移。",
    },
    "causal": {
        "lag_max_steps": 6,
        "min_signature_hit": 0.5,
        "primary_tol_steps": 1,
        "alpha_sigma": 1.0,
        "tie_break_proximity": True,
        "note": "声明的因果模型：每个根因一条**有序签名**。签名只声明，是否真的出现由事件流决定。",
    },
    "acceptance": {
        "min_anomaly_recall": 1.0,
        "max_anomaly_false_positives": 0,
        "min_five_answer_rate": 0.8,
        "min_chain_exact_rate": 0.6,
        "max_offset_error_s": 0.05,
        "max_drift_error_s_per_s": 0.0005,
        "min_alignment_lift": 0.2,
        "min_state_coverage": 6,
        "min_resource_coverage": 7,
        "min_job_nodes": 20,
    },
    "expectations": {
        "alignment_is_necessary": True,
        "constant_clock_offset_suffices": True,
        "single_global_baseline_suffices": True,
        "declared_signatures_all_materialize": True,
        "declared_causal_model_attributes_uniquely": True,
        "all_injected_anomalies_detected": True,
        "no_false_positive_anomalies": True,
        "three_graphs_separately_queryable": True,
        "recovery_lands_on_last_clean_checkpoint": True,
    },
    "report": {
        "title": "TrainingGraphFactory —— 训练运行时的数字孪生",
        "records_preview": 12,
    },
}


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并。**列表整体替换而不是拼接** —— 事件来源与根因都是「这一版就是
    这几个」，拼接会让改一次配置就静默多出几个来源。"""
    out: dict[str, Any] = dict(base)
    for k, v in over.items():
        cur = out.get(k)
        if isinstance(cur, Mapping) and isinstance(v, Mapping):
            out[k] = deep_merge(cur, v)
        else:
            out[k] = v
    return out


def resolve_path(root: Path | str, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(root) / p


def load_config(path: Path | str | None = None, *, root: Path | str) -> dict[str, Any]:
    """读配置。附加 `_config_path` / `_root`，并在校验失败时**立刻报错**。"""
    root = Path(root)
    cfg = dict(DEFAULTS)
    if path is not None:
        p = Path(path)
        cfg = deep_merge(cfg, json.loads(p.read_text(encoding="utf-8")))
    else:
        default = root / "configs" / "default.json"
        if default.exists():
            cfg = deep_merge(cfg, json.loads(default.read_text(encoding="utf-8")))
        p = default
    cfg["_config_path"] = str(p)
    cfg["_root"] = str(root)
    validate_config(cfg)
    return cfg


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def validate_config(cfg: Mapping[str, Any]) -> None:
    """配置校验。**宁可在这里炸，也不要跑到一半用一个荒唐的口径出报告。**

    本项目里最容易犯的三类错：① 声明了一个 `SIGNALS` 里没有的信号名（于是它永远
    检不出东西，而报告看起来正常）；② 声明了一个根因种类却忘了给它签名（于是它
    永远归因不到，而「归因失败」被读成「模型不够好」）；③ 签名里写了一个
    `EVIDENCE_LAYER` 里没有的证据标签（于是这条签名**永远不可能被满足**，
    而它表现为「这一类根因从来没出现过」）。
    """
    run = cfg.get("run") or {}
    steps = int(run.get("steps", 0))
    _require(steps >= 10, f"run.steps 至少 10，得到 {steps}")
    for key in ("batch_size", "seq_len", "ckpt_every", "eval_every"):
        _require(int(run.get(key, 0)) > 0, f"run.{key} 必须为正，得到 {run.get(key)}")
    _require(0.0 < float(run.get("loss_decay", 0)) <= 1.0, "run.loss_decay 必须在 (0, 1]")
    _require(float(run.get("length_cv", -1)) >= 0.0, "run.length_cv 不能为负")
    _require(float(run.get("per_token_ms", -1)) >= 0.0, "run.per_token_ms 不能为负")

    # ---- 来源
    sources = list(cfg.get("sources") or ())
    _require(len(sources) >= 2, f"至少要 2 个事件来源，得到 {len(sources)}")
    ids: list[str] = []
    for s in sources:
        sid = str(s.get("id", ""))
        _require(bool(sid), "事件来源必须有 id")
        _require(sid not in ids, f"事件来源 id 重复：{sid}")
        ids.append(sid)
        kind = str(s.get("kind", ""))
        _require(kind in SOURCE_KINDS, f"来源 {sid} 的 kind 非法：{kind!r}（可选 {', '.join(SOURCE_KINDS)}）")
        hz = s.get("sample_hz")
        _require(hz is None or float(hz) > 0, f"来源 {sid} 的 sample_hz 必须为正或 null")
        _require(
            bool(s.get("carries_step", False)) == (kind == "job"),
            f"来源 {sid}：只有 job 类来源才允许 carries_step=True",
        )
        if kind == "job":
            _require(
                float(s.get("clock_offset_s", 0.0)) == 0.0 and float(s.get("drift_s_per_s", 0.0)) == 0.0,
                f"参考时钟 {sid} 的偏移与漂移必须为 0（它的偏移不可辨识）",
            )
    jobs = [s for s in sources if str(s.get("kind")) == "job"]
    _require(len(jobs) == 1, f"必须恰好有一个 job 类来源，得到 {len(jobs)}")

    # ---- 信标
    beacon = cfg.get("beacon") or {}
    every = int(beacon.get("every_steps", 0))
    _require(every > 0, f"beacon.every_steps 必须为正，得到 {every}")
    _require(float(beacon.get("jitter_s", -1)) >= 0.0, "beacon.jitter_s 不能为负")
    drop = float(beacon.get("drop_prob", -1))
    _require(0.0 <= drop < 1.0, f"beacon.drop_prob 必须在 [0, 1)，得到 {drop}")

    # ---- 因果模型：必须覆盖全部种类，签名标签必须已声明
    model = cfg.get("causal_model") or {}
    _require(bool(model), "causal_model 不能为空")
    for kind in ROOT_CAUSE_KINDS:
        _require(kind in model, f"causal_model 缺少根因种类：{kind}")
    for kind, spec in model.items():
        _require(kind in ROOT_CAUSE_KINDS, f"causal_model 里有一个没人用的种类：{kind}")
        ps = str(spec.get("primary_signal", ""))
        _require(ps in SIGNALS, f"{kind}.primary_signal 未声明：{ps!r}")
        _require(bool(str(spec.get("config_key", ""))), f"{kind} 必须声明 config_key")
        res = str(spec.get("resource", ""))
        _require(res in RESOURCES, f"{kind}.resource 非法：{res!r}（可选 {', '.join(RESOURCES)}）")
        sig = list(spec.get("signature") or ())
        _require(bool(sig), f"{kind} 的签名为空（它会永远归因不到）")
        for lab in sig:
            _require(
                str(lab) in EVIDENCE_LAYER,
                f"{kind} 的签名里有未声明的证据标签：{lab!r}（可选 {', '.join(sorted(EVIDENCE_LAYER))}）",
            )

    # ---- 注入实例
    roots = list(cfg.get("root_causes") or ())
    _require(bool(roots), "root_causes 不能为空")
    seen: list[str] = []
    for r in roots:
        rid = str(r.get("id", ""))
        _require(bool(rid), "根因必须有 id")
        _require(rid not in seen, f"根因 id 重复：{rid}")
        seen.append(rid)
        kind = str(r.get("kind", ""))
        _require(kind in ROOT_CAUSE_KINDS, f"根因 {rid} 的 kind 非法：{kind!r}")
        at = int(r.get("at_step", -1))
        _require(0 <= at < steps, f"根因 {rid} 的 at_step={at} 越界（共 {steps} 步）")
        _require(int(r.get("duration_steps", 0)) >= 1, f"根因 {rid} 的 duration_steps 必须 ≥ 1")
        _require(float(r.get("magnitude", 0)) > 0, f"根因 {rid} 的 magnitude 必须为正")
        _require(
            "config_key" not in r and "resource" not in r,
            f"根因 {rid} 不许自带 config_key / resource —— 那是 causal_model 的职责，"
            "写进注入项会让复原器抄答案",
        )

    # ---- 检出
    det = cfg.get("detect") or {}
    _require(float(det.get("k", 0)) > 0, "detect.k 必须为正")
    _require(float(det.get("mad_floor_ratio", -1)) >= 0.0, "detect.mad_floor_ratio 不能为负")
    _require(int(det.get("min_stratum_size", 0)) >= 2, "detect.min_stratum_size 至少 2")
    monitored = list(det.get("monitored") or ())
    _require(bool(monitored), "detect.monitored 不能为空")
    for m in monitored:
        _require(str(m) in SIGNALS, f"detect.monitored 里有未声明的信号：{m!r}")

    # ---- 对齐
    al = cfg.get("align") or {}
    for key in ("residual_tol_s", "drift_tol_s_per_s", "max_step_assign_error"):
        _require(float(al.get(key, -1)) >= 0.0, f"align.{key} 不能为负")
    _require(int(al.get("min_beacons", 0)) >= 2, "align.min_beacons 至少 2（一个点拟不出漂移）")

    # ---- 因果
    ca = cfg.get("causal") or {}
    _require(int(ca.get("lag_max_steps", 0)) >= 1, "causal.lag_max_steps 必须 ≥ 1")
    _require(0.0 < float(ca.get("min_signature_hit", 0)) <= 1.0, "causal.min_signature_hit 必须在 (0, 1]")

    # ---- 验收 / 期望
    acc = cfg.get("acceptance") or {}
    _require(0.0 <= float(acc.get("min_five_answer_rate", -1)) <= 1.0, "acceptance.min_five_answer_rate 必须在 [0, 1]")
    _require(0.0 <= float(acc.get("min_chain_exact_rate", -1)) <= 1.0, "acceptance.min_chain_exact_rate 必须在 [0, 1]")
    _require(float(acc.get("min_alignment_lift", -1)) >= 0.0, "acceptance.min_alignment_lift 不能为负")
    exp = cfg.get("expectations") or {}
    _require(bool(exp), "expectations 不能为空")
    for k, v in exp.items():
        _require(isinstance(v, bool), f"期望 {k} 必须是布尔（它是「这条设计假设成立吗」的声明）")

    _require(len(STATES) >= int(acc.get("min_state_coverage", 0)), "min_state_coverage 超过了声明状态数")
    _require(len(RESOURCES) >= int(acc.get("min_resource_coverage", 0)), "min_resource_coverage 超过了声明资源数")


# --------------------------------------------------------------------------- #
# 摘要
# --------------------------------------------------------------------------- #

_PROTOCOL_KEYS = ("detect", "align", "causal", "causal_model", "acceptance", "expectations")
_INPUT_KEYS = ("seed", "run", "sources", "beacon", "root_causes")


def protocol_digest(cfg: Mapping[str, Any]) -> str:
    """**口径**摘要。改门槛 / 改因果模型会变，改注入实例不会。"""
    parts = [f"{k}:{sha256_text(canonical_json(cfg.get(k)))}" for k in _PROTOCOL_KEYS]
    return join_digest(parts)


def version_digest(cfg: Mapping[str, Any]) -> str:
    """**输入**摘要。改注入实例会变 —— 于是「版本号相同 ⇒ 这一跑逐位相同」。"""
    parts = [f"{k}:{sha256_text(canonical_json(cfg.get(k)))}" for k in _INPUT_KEYS]
    return join_digest(parts)


# --------------------------------------------------------------------------- #
# 访问器
# --------------------------------------------------------------------------- #


def sources_of(cfg: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(cfg.get("sources") or ())


def root_causes_of(cfg: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """按 `at_step` 排序。**排序是必要的**：模拟器按这个顺序施加影响，
    顺序变了「两个根因重叠」的形态就变了。"""
    return tuple(sorted(cfg.get("root_causes") or (), key=lambda r: (int(r["at_step"]), str(r["id"]))))


def causal_model_of(cfg: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(k): v for k, v in (cfg.get("causal_model") or {}).items()}


def signatures_of(cfg: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    return {
        str(k): tuple(str(x) for x in (v.get("signature") or ()))
        for k, v in causal_model_of(cfg).items()
    }


def expectations_of(cfg: Mapping[str, Any]) -> dict[str, bool]:
    return {str(k): bool(v) for k, v in (cfg.get("expectations") or {}).items()}
