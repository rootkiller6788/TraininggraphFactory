"""对抗测试。**不得依赖 `out/` 里的产物。**

理由与别处一样、但这里更硬：一个「读产物再断言产物」的对抗测试，测的是
「文件写出来了没有」，而不是「代码在遇到坏输入时会怎样」。产物是**结论**，
对抗要测的是**规则**。所以本模块只做两件事：

1. **拒收**：把明显错的配置/事件/状态喂进去，看它是不是**当场炸**。
   一个「悄悄接受」的校验比没有校验更糟 —— 它会让人以为这份配置被检查过了。
2. **决定论**：同一份输入跑两次逐位相同；把输入顺序打乱，结论不变。
   工作区的 git 兜底只到旧提交，这条判据是唯一能**当场**挡住「结果随 `set` 迭代顺序漂移」的东西。
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Mapping, Sequence

from .align import AlignmentError, align
from .config import DEFAULTS, deep_merge, validate_config
from .detect import detect
from .events import simulate
from .jobdag import GraphError, build_job_dag_from_events
from .recover import recover
from .rng import canonical_json, derive_seed
from .schema import EVIDENCE_LAYER, Event, StepSignal, evidence_hop, hop_key
from .state import build_state_graph
from .stats import step_kind, stratified_baselines

__all__ = ["AdversarialResult", "run_adversarial"]


class AdversarialResult:
    __slots__ = ("name", "passed", "detail")

    def __init__(self, name: str, passed: bool, detail: str = "") -> None:
        self.name = name
        self.passed = passed
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def _expect_raise(fn: Callable[[], Any], exc: type[BaseException]) -> tuple[bool, str]:
    try:
        fn()
    except exc as e:
        return True, f"{type(e).__name__}: {str(e)[:90]}"
    except Exception as e:  # 抛了别的异常 —— 也算没通过，但要说清楚
        return False, f"抛的是 {type(e).__name__} 而不是 {exc.__name__}：{str(e)[:70]}"
    return False, "**没有报错**（悄悄接受了坏输入）"


def _base_cfg() -> dict[str, Any]:
    return copy.deepcopy(DEFAULTS)


def _nan_safe(obj: Any) -> Any:
    """把 NaN 换成字符串 `"nan"` 再进指纹。

    `canonical_json` **故意**不允许 NaN（指纹里出现一个 NaN 就没法比大小、没法排序），
    但事件流里真的有 NaN —— 它是「测到了 nan」这个事实，必须能被比较。
    于是这里把「比较」与「指纹」分成两件事：比较时 NaN 是一个值，指纹里不许有它。
    """
    if isinstance(obj, float) and math.isnan(obj):
        return "nan"
    if isinstance(obj, dict):
        return {k: _nan_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_safe(v) for v in obj]
    return obj


def _same(a: Any, b: Any) -> bool:
    return canonical_json(_nan_safe(a)) == canonical_json(_nan_safe(b))


def _mut(cfg: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    return deep_merge(copy.deepcopy(cfg), patch)


def run_adversarial() -> list[AdversarialResult]:
    cfg = _base_cfg()
    out: list[AdversarialResult] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        out.append(AdversarialResult(name, ok, detail))

    # ------------------------------------------------------------------ 拒收
    ok, d = _expect_raise(
        lambda: validate_config(_mut(cfg, {"root_causes": [{**cfg["root_causes"][0], "config_key": "train.lr"}]})),
        ValueError,
    )
    add("拒收：`root_causes` 里带 `config_key`（那会让复原器抄答案）", ok, d)

    ok, d = _expect_raise(
        lambda: validate_config(_mut(cfg, {"root_causes": [{**cfg["root_causes"][0], "resource": "GPU0"}]})),
        ValueError,
    )
    add("拒收：`root_causes` 里带 `resource`（同上）", ok, d)

    ok, d = _expect_raise(
        lambda: validate_config(
            _mut(cfg, {"sources": [*cfg["sources"], {"id": "job2", "kind": "job", "carries_step": True}]})
        ),
        ValueError,
    )
    add("拒收：两个 `job` 源（偏移不可辨识，只能有一个基准）", ok, d)

    ok, d = _expect_raise(
        lambda: validate_config(
            _mut(
                cfg,
                {
                    "sources": [
                        {"id": "job", "kind": "job", "clock_offset_s": 0.5, "carries_step": True},
                        *[s for s in cfg["sources"] if s["id"] != "job"],
                    ]
                },
            )
        ),
        ValueError,
    )
    add("拒收：`job` 源带非零偏移（坐标系必须钉在它上面）", ok, d)

    ok, d = _expect_raise(
        lambda: validate_config(
            _mut(cfg, {"causal_model": {**cfg["causal_model"], "lr_spike": {"primary_signal": "loss", "resource": "GPU0", "signature": ["no_such_label"]}}})
        ),
        ValueError,
    )
    add("拒收：签名里出现未声明的证据标签", ok, d)

    ok, d = _expect_raise(
        lambda: validate_config(_mut(cfg, {"detect": {**cfg["detect"], "monitored": ["loss", "not_a_signal"]}})),
        ValueError,
    )
    add("拒收：`detect.monitored` 里出现未声明的信号", ok, d)

    # ------------------------------------------------------------------ 异质事件
    ok, d = _expect_raise(
        lambda: build_job_dag_from_events(
            [
                Event("e1", "job", "run_begin", 0.0, None, {}),
                Event("e2", "metric", "sample", 1.0, 3, {"loss": 1.0}),  # 非 job 源却带步号
            ],
            n_steps=10,
        ),
        GraphError,
    )
    add("拒收：Job DAG 收到「非 job 源却带步号」的事件", ok, d)

    # ------------------------------------------------------------------ 状态图
    from .schema import StateTransition

    class _TransOnly:
        """只带 `transitions` 的壳。`build_state_graph` 要的是「一个能问出转移的东西」，
        不是一个完整的 run —— 于是这里可以只喂几条手写的转移。"""

        __slots__ = ("transitions",)

        def __init__(self, ts: Sequence[Any]) -> None:
            self.transitions = tuple(ts)

    ok, d = _expect_raise(
        lambda: build_state_graph(_TransOnly([StateTransition(1, "TRAINING", "MOON", 0.0, "x")])),
        Exception,
    )
    add("拒收：状态图收到未声明的状态名", ok, d)

    ok, d = _expect_raise(
        lambda: build_state_graph(_TransOnly([StateTransition(1, "PREPARING", "DONE", 0.0, "跳级")])),
        Exception,
    )
    add("拒收：状态图收到未声明的转移边（`PREPARING → DONE`）", ok, d)

    ok, d = _expect_raise(
        lambda: build_state_graph(
            _TransOnly(
                [
                    StateTransition(1, "PREPARING", "TRAINING", 0.0, "ok"),
                    StateTransition(2, "DONE", "TRAINING", 0.0, "起点对不上"),
                ]
            )
        ),
        Exception,
    )
    add("拒收：状态图里某一步的起点不是当时的状态", ok, d)

    # ------------------------------------------------------------------ 证据标签
    ok, d = _expect_raise(lambda: evidence_hop("config_change", 42), ValueError)
    add("拒收：`config_change` 证据不带 `config_key`", ok, d)

    ok, d = _expect_raise(lambda: evidence_hop("state_failed", 97), ValueError)
    add("拒收：`state_failed` 证据不带 `state_key`", ok, d)

    ok, d = _expect_raise(lambda: hop_key("no_such_layer", "x", 1), ValueError)
    add("拒收：`hop_key` 用了未声明的层名", ok, d)

    ok, d = _expect_raise(lambda: evidence_hop("no_such_label", 1), ValueError)
    add("拒收：`evidence_hop` 用了未声明的证据标签", ok, d)

    # ------------------------------------------------------------------ 决定论
    run = simulate(cfg)
    run2 = simulate(cfg)
    same_events = _same([e.to_dict() for e in run.events], [e.to_dict() for e in run2.events])
    add("决定论：两次 `simulate` 的事件流逐位相同", same_events, f"{len(run.events)} 个事件")

    job = build_job_dag_from_events(run.events, n_steps=cfg["run"]["steps"])
    shuffled = list(run.events)
    # 用**同一个 Rng 派生出来的**打乱 —— 不用 `random.shuffle`，它依赖全局种子。
    from .rng import Rng

    Rng(derive_seed(cfg["seed"], "adversarial", "shuffle")).shuffled(shuffled)
    job2 = build_job_dag_from_events(shuffled, n_steps=cfg["run"]["steps"])
    add(
        "决定论：打乱 job 事件输入顺序后，Job DAG 完全相同",
        job == job2,
        f"{len(job.nodes)} 节点 / {len(job.edges)} 边",
    )
    add(
        "决定论：`topo_order` 入度为 0 时按节点 id 排序（不依赖 `set` 迭代顺序）",
        job.topo_order() == job2.topo_order(),
        f"前两个：{job.topo_order()[:2]}",
    )

    al = align(run, cfg)
    al2 = align(run2, cfg)
    add(
        "决定论：两次对齐给出同一张逐步信号表",
        _same([s.to_dict() for s in al.step_table], [s.to_dict() for s in al2.step_table]),
        f"{len(al.step_table)} 步",
    )

    # ------------------------------------------------------------------ 事件流的形状
    non_job_with_step = [e.id for e in run.events if e.source != "job" and e.step is not None]
    add(
        "事件流：不带步号的源**强制** `step=None`（否则等于偷偷对齐了）",
        not non_job_with_step,
        f"越界事件 {len(non_job_with_step)} 条",
    )
    beacons = [e for e in run.events if e.kind == "sync"]
    add(
        "事件流：四路来源各有一串同步信标（对齐的唯一锚点）",
        len({e.source for e in beacons}) == len(cfg["sources"]) and len(beacons) > 0,
        f"{len(beacons)} 个信标 / {len({e.source for e in beacons})} 路来源",
    )

    # ------------------------------------------------------------------ NaN 与分层
    table = al.step_table
    nan_steps = [s.step for s in table if s.get("loss") is not None and isinstance(s.get("loss"), float) and math.isnan(s.get("loss"))]
    add(
        "NaN 留在逐步信号表里（「测到了 nan」≠「没测到」）",
        len(nan_steps) > 0,
        f"loss 为 nan 的步：{nan_steps}",
    )

    dcfg = cfg["detect"]
    bl = stratified_baselines(
        table,
        ["step_time_s"],
        float(dcfg["mad_floor_ratio"]),
        min_stratum_size=int(dcfg["min_stratum_size"]),
        fallback_kind="none",
        detrend=True,
        span=float(dcfg["detrend_span"]),
    )
    kinds = {k for _s, k in bl}
    mixed = [s.step for s in table if step_kind(s) == "ckpt+eval"]
    add(
        "分层：混合步（ckpt+eval）被**单独识别**出来，不与纯 ckpt 步混在一起",
        bool(mixed) and "ckpt+eval" not in kinds,
        f"混合步 {mixed}；`step_time_s` 上够格的分层 {sorted(kinds)}（`ckpt+eval` 不在其中 = 没被并进 ckpt）",
    )

    det = detect(table, cfg)
    fp_before = len(det.anomalies)
    unstrat_cfg = _mut(cfg, {"detect": {"stratify_by_step_type": False, "stratum_fallback": "all"}})
    fp_after = len(detect(table, unstrat_cfg).anomalies)
    add(
        "分层确实在压制假阳性（不分层会多出成批的 step_time / disk 类假阳性）",
        fp_after > fp_before,
        f"分层 {fp_before} 条 → 不分层 {fp_after} 条",
    )
    judged_mixed = [st for st in mixed if (st, "step_time_s") in det.z]
    add(
        "分层样本不足时**不判定**，不拿全量基线兜底（混合步只有 1 个样本）",
        not judged_mixed and det.skipped.get("step_time_s", 0) >= 1,
        f"被误判的混合步 {judged_mixed or '无'}；`step_time_s` 未判定步数 {det.skipped.get('step_time_s', 0)}",
    )

    # ------------------------------------------------------------------ 答案不抄答案
    rec = recover(run, table, det, cfg, al)
    truth_ids = {t.root_cause_id for t in run.truths}
    leaked = [
        a.anomaly_id
        for a in rec.attributions
        if any(tid and tid in a.anomaly_id for tid in truth_ids)
    ]
    add(
        "答案里不出现真值项编号（`covered_anomalies` 只装检出编号）",
        not leaked and all(not str(x).startswith("rc-") for x in rec.unattributed),
        f"归因 {len(rec.attributions)} 起，泄漏 {len(leaked)} 处",
    )

    return out
