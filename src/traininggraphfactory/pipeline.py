"""把已经各自跑通的九个模块编排成一条真的会写盘的流水线。

```
simulate → align → {jobdag, state, resource} → detect → recover → verify → lineage
```

**两处刻意的顺序**：

1. `verify` 排在 `recover` **之后**、写盘 **之前**。因为它是第三方重放：
   它驱动真实模拟器去跑「去掉某个根因」的对照运行。如果把它挪到读命令里，
   一次「看一眼报告」就会重跑 6 次模拟 —— 而且读命令写不写盘这件事就说不清了。
2. 三张图在 `align` **之后**、`detect` **之前**。Job DAG 本来只吃 `job` 事件，
   不需要对齐；但它需要 `n_steps`，而对齐才是「到底有多少步」的权威。
   让三张图共用同一个 `n_steps`，比让它们各自猜一个更安全。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .align import align
from .config import expectations_of, protocol_digest, version_digest
from .detect import detect
from .events import simulate
from .jobdag import build_job_dag
from .lineage import build_lineage, dumps as lineage_dumps
from .recover import recover
from .resource import build_resource_graph
from .schema import utcnow
from .state import build_state_graph
from .verify import verify

__all__ = [
    "OUT_FILES",
    "EXTRA_FILES",
    "Version",
    "PipelineResult",
    "run_pipeline",
    "write_outputs",
    "load_result",
]

#: `run` 写出的产物。
OUT_FILES = (
    "report.md",
    "job_dag.json",
    "state_graph.json",
    "resource_graph.json",
    "step_signals.csv",
    "alignment.json",
    "anomalies.json",
    "attributions.json",
    "truths.json",
    "verify.json",
    "expectations.json",
    "lineage.json",
    "lineage_chain.dot",
    "version.json",
)

#: `visualize` 单独写的产物。
EXTRA_FILES = ("report.html",)


@dataclass(frozen=True)
class Version:
    id: str
    digest: str
    created_at: str
    protocol_digest: str
    n_events: int
    n_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "digest": self.digest,
            "created_at": self.created_at,
            "protocol_digest": self.protocol_digest,
            "n_events": self.n_events,
            "n_steps": self.n_steps,
        }


@dataclass(frozen=True)
class PipelineResult:
    """一次完整分析的全部结果。**只读命令全部从这里取数。**"""

    cfg: Mapping[str, Any]
    run: Any
    alignment: Any
    job_dag: Any
    state_graph: Any
    resource_graph: Any
    detect_result: Any
    recover_result: Any
    verify_result: Any
    lineage: Any
    version: Version

    # -- 三张图：各自的独立查询入口 ----------------------------------------

    def job_nodes(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.job_dag.nodes)

    def job_node_at(self, step: int) -> tuple[str, ...]:
        from .jobdag import job_nodes_at

        return tuple(n.id for n in job_nodes_at(self.job_dag, step))

    def job_deps(self, node_id: str) -> tuple[str, ...]:
        return self.job_dag.deps(node_id)

    def state_at(self, step: int) -> str:
        return self.state_graph.state_at(step)

    def state_path(self) -> tuple[tuple[int, str], ...]:
        return self.state_graph.path

    def resource_pressure(self, resource: str) -> Any:
        return self.resource_graph.pressure_of(resource)

    def bottleneck_at(self, step: int) -> str | None:
        return self.resource_graph.bottleneck_at(step)

    # -- 归因：五问的独立查询入口 ------------------------------------------

    def attribution(self, kind: str) -> Any:
        for a in self.recover_result.attributions:
            if a.root_cause_kind == kind:
                return a
        raise KeyError(f"没有这个根因的归因：{kind!r}")

    def truth(self, kind: str) -> Any:
        for t in self.run.truths:
            if t.kind == kind:
                return t
        raise KeyError(f"真值里没有这个根因：{kind!r}")

    def five_answers(self, kind: str) -> dict[str, Any]:
        """五问的结构化答案。**答案里不许出现真值** —— 真值要另一个命令去问。"""
        a = self.attribution(kind)
        return {
            "what": a.what,
            "when": a.when_step,
            "resource": a.resource,
            "config": {"key": a.config_key, "value": a.config_value},
            "checkpoint": a.checkpoint,
            "confidence": a.confidence,
            "ambiguous": a.ambiguous,
            "chain": [h.to_dict() for h in a.chain],
        }


def run_pipeline(cfg: Mapping[str, Any]) -> PipelineResult:
    """端到端跑一遍。**纯计算，不写盘。**"""
    run = simulate(cfg)
    al = align(run, cfg)
    step_table = al.step_table

    job = build_job_dag(run)
    stg = build_state_graph(run)
    rg = build_resource_graph(step_table, cfg)

    det = detect(step_table, cfg)
    rec = recover(run, step_table, det, cfg, al)
    ver = verify(cfg, step_table, rec.attributions)

    lin = build_lineage(
        run=run,
        al=al,
        det=det,
        rec=rec,
        sources=[s.id for s in run.sources],
        n_steps=len(step_table),
        graphs={"job": len(job.nodes), "state": len(stg.nodes), "resource": len(rg.resources)},
    )

    digest = version_digest(cfg)
    version = Version(
        id=f"v1-{digest[:12]}",
        digest=digest,
        created_at=utcnow(),
        protocol_digest=protocol_digest(cfg),
        n_events=len(run.events),
        n_steps=len(step_table),
    )
    return PipelineResult(
        cfg=cfg,
        run=run,
        alignment=al,
        job_dag=job,
        state_graph=stg,
        resource_graph=rg,
        detect_result=det,
        recover_result=rec,
        verify_result=ver,
        lineage=lin,
        version=version,
    )


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


_STEP_SIGNAL_COLUMNS = (
    "step",
    "t_ref_begin",
    "n_samples",
    "n_sources",
    "kind",
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
    "lr",
)


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if v != v:  # NaN
            return "nan"
        return repr(v)
    return str(v)


def _csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = [",".join(columns)]
    for r in rows:
        lines.append(",".join(_cell(r.get(c)) for c in columns))
    return "\n".join(lines) + "\n"


def write_outputs(result: PipelineResult, out_dir: Path | str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # --- 三张图，三个文件。**分开写**，分开的产物才谈得上分开查询。 ---
    _write_json(out / "job_dag.json", result.job_dag.to_dict())
    written.append(out / "job_dag.json")
    _write_json(out / "state_graph.json", result.state_graph.to_dict())
    written.append(out / "state_graph.json")
    _write_json(out / "resource_graph.json", result.resource_graph.to_dict())
    written.append(out / "resource_graph.json")

    # --- 逐步信号表：**唯一**一张逐点的原始表，其余产物都是它的派生物。 ---
    from .stats import step_kind

    rows = []
    for st in result.alignment.step_table:
        rows.append(
            {
                "step": st.step,
                "t_ref_begin": st.t_ref_begin,
                "n_samples": st.n_samples,
                "n_sources": st.n_sources,
                "kind": step_kind(st),
                **{s: st.values.get(s) for s in st.values},
            }
        )
    (out / "step_signals.csv").write_text(_csv(rows, _STEP_SIGNAL_COLUMNS), encoding="utf-8")
    written.append(out / "step_signals.csv")

    # --- 对齐：三种假设同台，选中的那个要能解释为什么。 ---
    # 逐步信号表**只存一份**（`step_signals.csv`），这里只留步边界 ——
    # 同一张 120 步的表在两个文件里各存一份，就是两个会漂移的真相。
    _write_json(
        out / "alignment.json",
        {
            "reports": [r.to_dict() for r in result.alignment.reports],
            "candidates": [
                {"source": s, "models": [m.to_dict() for m in ms]}
                for s, ms in result.alignment.candidates
            ],
            "chosen_model": dict(result.alignment.chosen_model),
            "step_boundaries": [
                {"step": s, "t_ref_begin": a, "t_ref_end": b}
                for s, a, b in result.alignment.step_boundaries
            ],
            "n_events_total": result.alignment.n_events_total,
            "n_events_assigned": result.alignment.n_events_assigned,
        },
    )
    written.append(out / "alignment.json")

    # --- 检出 ---
    _write_json(out / "anomalies.json", result.detect_result.to_dict())
    written.append(out / "anomalies.json")

    # --- 归因（答案）与真值（答案的判据）**分成两个文件**。 ---
    _write_json(
        out / "attributions.json",
        {
            "attributions": [a.to_dict() for a in result.recover_result.attributions],
            "unattributed": list(result.recover_result.unattributed),
            "n_anomalies": result.recover_result.n_anomalies,
        },
    )
    written.append(out / "attributions.json")

    _write_json(
        out / "truths.json",
        {
            "truths": [t.to_dict() for t in result.run.truths],
            "n_roots": len(result.run.truths),
        },
    )
    written.append(out / "truths.json")

    # --- 校验（第三方） ---
    _write_json(out / "verify.json", result.verify_result.to_dict())
    written.append(out / "verify.json")

    # --- 设计期望 vs 实测：**允许期望被推翻**。 ---
    from .report import expectation_rows

    _write_json(
        out / "expectations.json",
        {
            "declared": dict(expectations_of(result.cfg)),
            "rows": expectation_rows(result),
        },
    )
    written.append(out / "expectations.json")

    # --- 血缘 ---
    (out / "lineage.json").write_text(lineage_dumps(result.lineage), encoding="utf-8")
    written.append(out / "lineage.json")
    keep = {
        n.id
        for n in result.lineage.nodes
        if n.kind in ("run", "stream", "stage", "graph", "anomaly", "hop", "attribution", "answer")
    }
    (out / "lineage_chain.dot").write_text(result.lineage.to_dot(include=sorted(keep)), encoding="utf-8")
    written.append(out / "lineage_chain.dot")

    _write_json(out / "version.json", result.version.to_dict())
    written.append(out / "version.json")

    from .report import render_report

    (out / "report.md").write_text(render_report(result), encoding="utf-8")
    written.append(out / "report.md")
    return written


def load_result(out_dir: Path | str) -> PipelineResult | None:
    """从 `out/` 重建结果。缺任何一个产物都返回 `None`（**不抛异常、不重跑**）。

    返回 `None` 而不是抛异常，是为了让 `cli` 能干净地决定「退回在内存里跑一遍」。
    一个只读命令因为「少了一个文件」而崩掉，用户会以为工具坏了。

    注意：`verify_result` 是**唯一**没法从产物完整重建的东西 —— 探针明细在
    `verify.json` 里，但 `alpha_sigma` 之类的口径参数要从配置取。所以这里
    把配置一起读回来，两边对齐。
    """
    out = Path(out_dir)
    if not out.is_dir():
        return None
    needed = (
        "job_dag.json",
        "state_graph.json",
        "resource_graph.json",
        "step_signals.csv",
        "alignment.json",
        "anomalies.json",
        "attributions.json",
        "truths.json",
        "verify.json",
        "lineage.json",
        "version.json",
    )
    for name in needed:
        if not (out / name).is_file():
            return None

    from .config import load_config
    from .schema import (
        AlignmentModel,
        AlignmentReport,
        AlignmentResult,
        Anomaly,
        Attribution,
        JobDag,
        ResourceGraph,
        StateGraph,
        StepSignal,
        TruthRecord,
        VerifyResult,
    )
    from .lineage import Lineage
    from .stats import Baseline

    root = out.parent
    cfg = load_config(root=root)

    version_raw = json.loads((out / "version.json").read_text(encoding="utf-8"))
    version = Version(
        id=str(version_raw["id"]),
        digest=str(version_raw["digest"]),
        created_at=str(version_raw["created_at"]),
        protocol_digest=str(version_raw["protocol_digest"]),
        n_events=int(version_raw["n_events"]),
        n_steps=int(version_raw["n_steps"]),
    )

    # --- 逐步信号表 ---
    step_table = _read_step_signals(out / "step_signals.csv")

    # --- 对齐 ---
    al_raw = json.loads((out / "alignment.json").read_text(encoding="utf-8"))
    alignment = AlignmentResult.from_dict(
        {
            **al_raw,
            "step_table": [st.to_dict() for st in step_table],
        }
    )

    # --- 三张图 ---
    job_dag = JobDag.from_dict(json.loads((out / "job_dag.json").read_text(encoding="utf-8")))
    state_graph = StateGraph.from_dict(json.loads((out / "state_graph.json").read_text(encoding="utf-8")))
    resource_graph = ResourceGraph.from_dict(
        json.loads((out / "resource_graph.json").read_text(encoding="utf-8"))
    )

    # --- 检出 ---
    det_raw = json.loads((out / "anomalies.json").read_text(encoding="utf-8"))
    det = _DetectShell(
        anomalies=tuple(Anomaly.from_dict(a) for a in det_raw["anomalies"]),
        z={},
        baselines={
            (str(b["key"]).split("@")[0], str(b["key"]).split("@")[1]): Baseline.from_dict(b)
            for b in det_raw.get("baselines", [])
        },
        threshold=float(det_raw["threshold"]),
        monitored=tuple(str(s) for s in det_raw["monitored"]),
        skipped={str(k): int(v) for k, v in det_raw.get("skipped_steps", {}).items()},
    )

    # --- 归因 ---
    at_raw = json.loads((out / "attributions.json").read_text(encoding="utf-8"))
    rec = _RecoverShell(
        attributions=tuple(Attribution.from_dict(a) for a in at_raw["attributions"]),
        unattributed=tuple(str(x) for x in at_raw["unattributed"]),
        n_anomalies=int(at_raw["n_anomalies"]),
    )

    # --- 真值 ---
    tr_raw = json.loads((out / "truths.json").read_text(encoding="utf-8"))
    truths = tuple(TruthRecord.from_dict(t) for t in tr_raw["truths"])

    # --- 校验 ---
    ver = VerifyResult.from_dict(json.loads((out / "verify.json").read_text(encoding="utf-8")))

    # --- 血缘 ---
    lin = Lineage.from_dict(json.loads((out / "lineage.json").read_text(encoding="utf-8")))

    run = _RunShell(sources=_sources_from_cfg(cfg), truths=truths, events=())

    return PipelineResult(
        cfg=cfg,
        run=run,
        alignment=alignment,
        job_dag=job_dag,
        state_graph=state_graph,
        resource_graph=resource_graph,
        detect_result=det,
        recover_result=rec,
        verify_result=ver,
        lineage=lin,
        version=version,
    )


# --------------------------------------------------------------------------- #
# 只读外壳：产物重建出来的对象只支持**查询**，不支持再跑流水线
# --------------------------------------------------------------------------- #


class _DetectShell:
    """`DetectResult` 的只读外壳。`by_signal` / `at_step` 的语义与真货一致。"""

    __slots__ = ("anomalies", "z", "baselines", "threshold", "monitored", "skipped")

    def __init__(self, **kw: Any) -> None:
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def by_signal(self) -> dict[str, tuple[Any, ...]]:
        out: dict[str, list[Any]] = {}
        for a in self.anomalies:
            out.setdefault(a.signal, []).append(a)
        return {k: tuple(v) for k, v in out.items()}

    def at_step(self, step: int) -> tuple[Any, ...]:
        return tuple(a for a in self.anomalies if a.step_lo <= step <= a.step_hi)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomalies": [a.to_dict() for a in self.anomalies],
            "threshold": self.threshold,
            "monitored": list(self.monitored),
            "skipped_steps": dict(self.skipped),
        }


class _RecoverShell:
    __slots__ = ("attributions", "unattributed", "n_anomalies")

    def __init__(self, **kw: Any) -> None:
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def attribution(self, key: str) -> Any:
        for a in self.attributions:
            if a.root_cause_kind == key:
                return a
        raise KeyError(f"没有这个根因的归因：{key!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "attributions": [a.to_dict() for a in self.attributions],
            "unattributed": list(self.unattributed),
            "n_anomalies": self.n_anomalies,
        }


class _RunShell:
    """`SimRun` 的只读外壳。**只带真值与来源清单** —— 事件流是 2000 条，
    为了看一份报告把它们反序列化出来不值当；真要事件流的命令走「内存重跑」。"""

    __slots__ = ("sources", "truths", "events")

    def __init__(self, **kw: Any) -> None:
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def events_of(self, source_id: str) -> tuple[Any, ...]:
        return tuple(e for e in self.events if e.source == source_id)

    def trace(self, step: int) -> Any:
        return None


def _sources_from_cfg(cfg: Mapping[str, Any]) -> tuple[Any, ...]:
    from .schema import SourceSpec

    return tuple(
        SourceSpec(
            id=str(s["id"]),
            kind=str(s.get("kind", s["id"])),
            clock_offset_s=float(s.get("clock_offset_s", 0.0)),
            drift_s_per_s=float(s.get("drift_s_per_s", 0.0)),
            sample_hz=(float(s["sample_hz"]) if s.get("sample_hz") is not None else None),
            carries_step=bool(s.get("carries_step", False)),
            note=str(s.get("note", "")),
        )
        for s in cfg.get("sources", ())
    )


def _read_step_signals(path: Path) -> tuple[Any, ...]:
    """读回逐步信号表。**`nan` 与空串必须区分开**：前者是「测到了 nan」，
    后者是「这一步没测到」。CSV 里 `nan` 与 `` 就是这两个不同的东西。"""
    from .schema import StepSignal

    text = path.read_text(encoding="utf-8").strip("\n")
    lines = text.split("\n")
    header = lines[0].split(",")
    idx = {c: i for i, c in enumerate(header)}
    value_cols = [c for c in header if c not in ("step", "t_ref_begin", "n_samples", "n_sources", "kind")]
    out: list[Any] = []
    for ln in lines[1:]:
        if not ln:
            continue
        cells = ln.split(",")
        vals: dict[str, float | None] = {}
        for c in value_cols:
            raw = cells[idx[c]]
            if raw == "":
                vals[c] = None
            elif raw == "nan":
                vals[c] = float("nan")
            else:
                vals[c] = float(raw)
        out.append(
            StepSignal(
                step=int(cells[idx["step"]]),
                t_ref_begin=float(cells[idx["t_ref_begin"]]),
                values=vals,
                n_samples=int(cells[idx["n_samples"]]),
                n_sources=int(cells[idx["n_sources"]]),
            )
        )
    return tuple(out)
