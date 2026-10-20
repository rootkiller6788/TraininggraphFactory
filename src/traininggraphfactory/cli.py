"""命令行。**读命令不写盘**，写命令只有 `run` / `visualize`。

这不是洁癖：`out/` 里那份结果可能对应另一份配置，用户看一眼报告就会拿它下结论。
如果「看一眼」会顺手改写产物，结论与配置的对应关系就断了。所以：

- 读命令**从产物重建结果对象**（`pipeline.load_result`），绝不重跑流水线；
- 产物缺失时，读命令在内存里跑一遍 —— 仍然**不落盘**，并明确告诉你「本次是内存重跑」；
- `acceptance` 的**对抗测试**根本不读产物（`acceptance.py` 只吃配置与合成事件），
  于是「代码在遇到坏输入时会怎样」不会因为产物过期而失真。

命令清单（前 13 个只读）：

```
summary / graphs / job / state / resource / align / anomalies / attribution
/ truth / verify / expectations / acceptance / lineage
run / visualize          ← 写
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .acceptance import run_adversarial
from .config import load_config
from .events import dump_events, load_events
from .pipeline import EXTRA_FILES, OUT_FILES, load_result, run_pipeline, write_outputs
from .recover import group_key
from .report import acceptance_checks, expectation_rows, fmt

#: 项目根目录（本文件在 `src/traininggraphfactory/cli.py`）。
ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DEFAULT_CONFIG = os.path.join("configs", "default.json")

#: 只读命令。**一个都不许写盘** —— 验收脚本会逐条核对。
READ_ONLY = (
    "summary",
    "graphs",
    "job",
    "state",
    "resource",
    "align",
    "anomalies",
    "attribution",
    "truth",
    "verify",
    "expectations",
    "acceptance",
    "lineage",
    "events",
)

WRITE = ("run", "visualize")


def _out_dir(args: argparse.Namespace) -> Path:
    p = Path(args.out)
    return p if p.is_absolute() else ROOT / p


def _data_dir(args: argparse.Namespace) -> Path:
    """事件流落地的目录，**钉在项目根上，不跟 `--out` 走**。

    `--out` 选的是「结论放哪」，不是「输入放哪」。让采集跟着 `--out` 走，
    会得到一个「同一份输入被写进五个目录」的局面 —— 而这份 JSONL 的意义
    恰恰是「这一份输入，被这一份结论消费」。
    """
    return ROOT / "data"


# ------------------------------------------------------------ --json 的出口

#: 真正的 stdout。`--json` 时命令里的 `print` 会被 `_quiet_prose` 吞掉。
_REAL_STDOUT = sys.stdout


def _emit_json(payload: Any) -> None:
    """`--json` 的唯一出口：**只**打印一段 JSON，不夹任何说明文字。"""
    _REAL_STDOUT.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _REAL_STDOUT.flush()


class _quiet_prose:
    """把命令里的 `print` 全部吞掉，只在 `--json` 时使用。"""

    def __enter__(self) -> "_quiet_prose":
        import io

        self._saved = sys.stdout
        sys.stdout = io.StringIO()
        return self

    def __exit__(self, *exc: Any) -> bool:
        sys.stdout = self._saved
        return False


def _w(s: Any) -> int:
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def _pad(s: Any, width: int, *, right: bool = False) -> str:
    gap = " " * max(0, width - _w(s))
    return (gap + str(s)) if right else (str(s) + gap)


def _row(cells: Sequence[tuple[str, int]], *, right: Sequence[int] = ()) -> str:
    parts = [_pad(v, w, right=i in right) for i, (v, w) in enumerate(cells)]
    return "  " + " ".join(parts).rstrip()


# ---------------------------------------------------------------------- 取数


def _result(args: argparse.Namespace) -> tuple[Any, bool]:
    """返回 `(结果, 是否来自产物)`。**永不落盘。**"""
    cached = load_result(_out_dir(args))
    if cached is not None:
        return cached, True
    cfg = load_config(args.config, root=ROOT)
    return run_pipeline(cfg), False


def _note(from_products: bool) -> str:
    return "" if from_products else "  （产物不存在：本次在内存中重跑，未写盘）"


# ---------------------------------------------------------------------- 命令


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, root=ROOT)
    result = run_pipeline(cfg)
    written = write_outputs(result, _out_dir(args))
    captured = ""
    if not args.no_capture:
        # M0「跑通采集一次」：把四路事件流落成 JSONL。
        # **它写进 `data/`，不进 `out/`** —— `out/` 装的是「分析的结论」，
        # 而这里是「分析的输入」。混在一起，`load_result` 就分不清
        # 「这份结论对应哪份输入」了。
        captured = str(dump_events(result.run.events, _data_dir(args) / "run" / "events.jsonl"))
    print(f"跑了 {result.version.n_steps} 步 / {result.version.n_events} 个事件")
    print(f"三张图：job {len(result.job_dag.nodes)} 节点、state {len(result.state_graph.visited_nodes)} 状态、"
          f"resource {len(result.resource_graph.resources)} 资源")
    print(f"检出 {len(result.detect_result.anomalies)} 条 → 归因 {len(result.recover_result.attributions)} 起，"
          f"未归因 {len(result.recover_result.unattributed)} 条")
    acc = acceptance_checks(result)
    n_ok = sum(1 for a in acc if a["passed"])
    print(f"验收 {n_ok}/{len(acc)}；版本 {result.version.id}")
    print(f"写出 {len(written)} 个产物 → {_out_dir(args)}")
    if captured:
        print(f"采集事件流 → {captured}")
    if args.json:
        _emit_json(
            {
                "version": result.version.to_dict(),
                "written": [str(p) for p in written],
                "events_captured": captured,
                "n_anomalies": len(result.detect_result.anomalies),
                "n_attributions": len(result.recover_result.attributions),
                "acceptance": {"passed": n_ok, "total": len(acc)},
            }
        )
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    from .visualize import render_html

    out = _out_dir(args)
    cached = load_result(out)
    if cached is not None:
        result = cached
    else:
        result = run_pipeline(load_config(args.config, root=ROOT))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.html"
    path.write_text(render_html(result), encoding="utf-8")
    print(f"写出 {path}")
    if args.json:
        _emit_json({"written": str(path), "bytes": path.stat().st_size})
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    result, cached = _result(args)
    acc = acceptance_checks(result)
    n_ok = sum(1 for a in acc if a["passed"])
    if args.json:
        _emit_json(
            {
                "from_products": cached,
                "version": result.version.to_dict(),
                "n_steps": len(result.alignment.step_table),
                "graphs": {
                    "job": {"nodes": len(result.job_dag.nodes), "edges": len(result.job_dag.edges)},
                    "state": {
                        "declared": len(result.state_graph.nodes),
                        "visited": len(result.state_graph.visited_nodes),
                        "unvisited": list(result.state_graph.unvisited_nodes),
                    },
                    "resource": {
                        "resources": list(result.resource_graph.resources),
                        "bottlenecks": len(result.resource_graph.bottleneck_by_step),
                    },
                },
                "anomalies": len(result.detect_result.anomalies),
                "attributions": len(result.recover_result.attributions),
                "unattributed": list(result.recover_result.unattributed),
                "acceptance": {"passed": n_ok, "total": len(acc)},
            }
        )
        return 0
    print(f"版本 {result.version.id} ｜ 口径 {result.version.protocol_digest[:12]}{_note(cached)}")
    print(f"步数 {len(result.alignment.step_table)} ｜ 事件 {result.version.n_events}")
    print()
    print("三张图（**分开构建、分开查询**）")
    print(_row([("Job DAG", 12), (f"{len(result.job_dag.nodes)} 节点 / {len(result.job_dag.edges)} 边", 24),
                (f"模板 {len(result.job_dag.template_nodes)} 节点", 20)]))
    print(_row([("State", 12), (f"{len(result.state_graph.visited_nodes)}/{len(result.state_graph.nodes)} 状态走过", 24),
                (f"没走到：{', '.join(result.state_graph.unvisited_nodes) or '（无）'}", 20)]))
    print(_row([("Resource", 12), (f"{len(result.resource_graph.resources)} 个资源", 24),
                (f"瓶颈 {len(result.resource_graph.bottleneck_by_step)} 步", 20)]))
    print()
    print(f"检出 {len(result.detect_result.anomalies)} 条 → 归因 {len(result.recover_result.attributions)} 起，"
          f"未归因 {len(result.recover_result.unattributed)} 条")
    print(f"验收 {n_ok}/{len(acc)}")
    return 0


def cmd_graphs(args: argparse.Namespace) -> int:
    result, cached = _result(args)
    job, stg, rg = result.job_dag, result.state_graph, result.resource_graph
    if args.json:
        _emit_json(
            {
                "from_products": cached,
                "job": job.to_dict(),
                "state": stg.to_dict(),
                "resource": rg.to_dict(),
            }
        )
        return 0
    print(f"三张图{_note(cached)}")
    print()
    print(f"[1] Job DAG：{len(job.nodes)} 实例节点 / {len(job.edges)} 实例边")
    print(f"[2] State Graph：{len(stg.nodes)} 状态 / {len(stg.edges)} 边，走过 {len(stg.visited_nodes)}")
    print(f"[3] Resource Graph：中枢 {rg.hub}，{len(rg.resources)} 资源")
    print()
    print("**三张图各自有独立产物**：job_dag.json / state_graph.json / resource_graph.json")
    return 0


def cmd_job(args: argparse.Namespace) -> int:
    result, cached = _result(args)
    job = result.job_dag
    if args.node:
        if not job.has(args.node):
            print(f"没有这个节点：{args.node}", file=sys.stderr)
            return 2
        deps = job.deps(args.node)
        anc = job.ancestors(args.node)
        if args.json:
            _emit_json(
                {
                    "from_products": cached,
                    "node": job.node(args.node).to_dict(),
                    "deps": list(deps),
                    "ancestors": list(anc),
                    "n_ancestors": len(anc),
                }
            )
            return 0
        n = job.node(args.node)
        print(f"{n.id}  [{n.template}]  步 {n.step_lo}-{n.step_hi}  事件 {n.n_events}")
        print(f"依赖：{', '.join(deps) or '（无）'}")
        print(f"祖先 {len(anc)} 个：{', '.join(anc[:12])}{' …' if len(anc) > 12 else ''}")
        return 0
    if args.step is not None:
        ids = result.job_node_at(args.step)
        if args.json:
            _emit_json({"from_products": cached, "step": args.step, "nodes": list(ids)})
            return 0
        print(f"第 {args.step} 步的 job 节点：{', '.join(ids) or '（无）'}")
        return 0
    if args.json:
        _emit_json({"from_products": cached, **job.to_dict(), "topo_order": list(job.topo_order())})
        return 0
    print(f"Job DAG：{len(job.nodes)} 实例节点 / {len(job.edges)} 实例边 ｜ "
          f"模板 {len(job.template_nodes)} 节点 / {len(job.template_edges)} 边{_note(cached)}")
    print()
    counts: dict[str, int] = {}
    for n in job.nodes:
        counts[n.template] = counts.get(n.template, 0) + 1
    for t, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(_row([(t, 16), (f"×{c}", 8)]))
    print()
    order = job.topo_order()
    print(f"拓扑序前两个：{order[0]} → {order[1]}（入度为 0 时按节点 id 排序取下一个）")
    if job.has("checkpoint@72"):
        print(f"checkpoint@72 依赖：{', '.join(job.deps('checkpoint@72'))}")
        print(f"checkpoint@72 祖先 {len(job.ancestors('checkpoint@72'))} 个")
    return 0


def cmd_state(args: argparse.Namespace) -> int:
    result, cached = _result(args)
    stg = result.state_graph
    if args.step is not None:
        s = stg.state_at(args.step)
        if args.json:
            _emit_json({"from_products": cached, "step": args.step, "state": s,
                        "transitions": [t.to_dict() for t in stg.transitions_at(args.step)]})
            return 0
        print(f"第 {args.step} 步：{s}")
        for t in stg.transitions_at(args.step):
            print(f"  {t.frm} → {t.to}  ({t.reason})")
        return 0
    if args.json:
        _emit_json({"from_products": cached, **stg.to_dict()})
        return 0
    print(f"State Graph：声明 {len(stg.nodes)} 状态 / {len(stg.edges)} 边 ｜ "
          f"走过 {len(stg.visited_nodes)} 状态 / {len(stg.visited_edges)} 边{_note(cached)}")
    print()
    print(f"路径：{' → '.join(s for _st, s in stg.path)}")
    print(f"没走到的状态：{', '.join(stg.unvisited_nodes) or '（无）'}")
    print(f"没走到的转移：{'、'.join(f'{a}→{b}' for a, b in stg.unvisited_edges) or '（无）'}")
    print()
    for s, n in sorted(stg.steps_in_state, key=lambda kv: (-kv[1], kv[0])):
        print(_row([(s, 16), (str(n), 6)], right=(1,)))
    return 0


def cmd_resource(args: argparse.Namespace) -> int:
    result, cached = _result(args)
    rg = result.resource_graph
    if args.step is not None:
        b = rg.bottleneck_at(args.step)
        if args.json:
            _emit_json({"from_products": cached, "step": args.step, "bottleneck": b})
            return 0
        print(f"第 {args.step} 步瓶颈：{b or '（无显著压力）'}")
        return 0
    if args.resource:
        p = rg.pressure_of(args.resource)
        if args.json:
            _emit_json({"from_products": cached, "pressure": p.to_dict()})
            return 0
        print(f"{p.resource}（{rg.signal_of.get(p.resource, '')}）：峰值 {fmt(p.peak_value)} @{p.peak_step}，"
              f"基线 {fmt(p.baseline)}，归一化 z={fmt(p.normalized, sig=3)}，方向 {p.direction}")
        return 0
    if args.json:
        _emit_json({"from_products": cached, **rg.to_dict()})
        return 0
    print(f"Resource Graph：中枢 {rg.hub}，{len(rg.resources)} 个资源{_note(cached)}")
    print()
    print(_row([("资源", 10), ("信号", 14), ("峰值步", 8), ("峰值", 12), ("基线", 12), ("z", 8), ("方向", 10)], right=(2, 3, 4, 5)))
    for p in rg.ranked():
        print(_row([(p.resource, 10), (rg.signal_of.get(p.resource, ""), 14), (str(p.peak_step), 8),
                    (fmt(p.peak_value), 12), (fmt(p.baseline), 12), (fmt(p.normalized, sig=3), 8),
                    (p.direction, 10)], right=(2, 3, 4, 5)))
    print()
    print(f"瓶颈只在压力超过 k 时报，本次 {len(rg.bottleneck_by_step)} 步")
    return 0


def cmd_align(args: argparse.Namespace) -> int:
    result, cached = _result(args)
    al = result.alignment
    if args.json:
        _emit_json(
            {
                "from_products": cached,
                "reports": [r.to_dict() for r in al.reports],
                "candidates": [
                    {"source": s, "models": [m.to_dict() for m in ms]} for s, ms in al.candidates
                ],
                "chosen_model": dict(al.chosen_model),
                "n_events_total": al.n_events_total,
                "n_events_assigned": al.n_events_assigned,
            }
        )
        return 0
    print(f"对齐：{al.n_events_total} 个待归属事件 → {al.n_events_assigned} 个接上了步{_note(cached)}")
    print()
    for r in al.reports:
        print(f"[{r.source}] 选中 {r.model}")
        by_name = {m.model: m for m in al.models_of(r.source)}
        for name in ("identity", "constant", "linear"):
            m = by_name.get(name)
            if m is None:
                continue
            mark = " ←" if r.model == name else ""
            print(_row([("", 4), (name + mark, 12), (f"a={fmt(m.a, sig=5)}", 14), (f"b={fmt(m.b, sig=6)}", 16),
                        (f"resid={fmt(m.residual_rms_s, sig=3)}", 16),
                        (f"err={m.assign_error_rate * 100:.2f}%", 12)]))
        print(_row([("", 4), ("误差", 12), (f"偏移 {abs(r.estimated_offset_s - r.injected_offset_s):.5f}s", 20),
                    (f"漂移 {abs(r.estimated_drift_s_per_s - r.injected_drift_s_per_s):.7f}", 20)]))
    return 0


def cmd_anomalies(args: argparse.Namespace) -> int:
    result, cached = _result(args)
    det = result.detect_result
    items = det.by_signal().get(args.signal, ()) if args.signal else det.anomalies
    if args.json:
        _emit_json(
            {
                "from_products": cached,
                "n": len(items),
                "threshold": det.threshold,
                "skipped": dict(det.skipped),
                "anomalies": [a.to_dict() for a in items],
            }
        )
        return 0
    print(f"检出 {len(items)} 条（k={fmt(det.threshold, sig=2)}）{_note(cached)}")
    if det.skipped:
        print(f"未判定（分层样本不足）：{len(det.skipped)} 个分层 —— 「未判定」与「正常」必须分开报")
    print()
    print(_row([("id", 22), ("信号", 14), ("步", 12), ("峰值 z", 10), ("NaN", 6)]))
    for a in items:
        span = f"{a.step_lo}-{a.step_hi}" if a.step_hi > a.step_lo else str(a.step_lo)
        print(_row([(a.id, 22), (a.signal, 14), (span, 12), (fmt(a.peak_z, sig=3), 10),
                    ("是" if a.is_nan else "否", 6)]))
    return 0


def cmd_attribution(args: argparse.Namespace) -> int:
    result, cached = _result(args)
    rec = result.recover_result
    items = [rec.attribution(args.kind)] if args.kind else list(rec.attributions)
    if args.json:
        payload: Any
        if args.kind:
            payload = {"from_products": cached, **result.five_answers(args.kind)}
        else:
            payload = {
                "from_products": cached,
                "attributions": [
                    {"key": group_key(a), **result.five_answers(a.root_cause_kind)} for a in items
                ],
                "unattributed": list(rec.unattributed),
            }
        _emit_json(payload)
        return 0
    print(f"归因 {len(rec.attributions)} 起，未归因检出 {len(rec.unattributed)} 条{_note(cached)}")
    print()
    for a in items:
        ans = result.five_answers(a.root_cause_kind)
        print(f"[{group_key(a)}]  {a.root_cause_kind}  置信度 {fmt(a.confidence, sig=2)}"
              f"{'  有歧义' if a.ambiguous else ''}")
        print(f"    what      : {ans['what']}")
        print(f"    when      : 第 {ans['when']} 步")
        print(f"    resource  : {ans['resource']}")
        print(f"    config    : {ans['config']['key']} = {fmt(ans['config']['value'])}")
        print(f"    checkpoint: {ans['checkpoint'] or '—'}")
        print("    chain     : " + " → ".join(h["key"] for h in ans["chain"]))
        print()
    if rec.unattributed:
        print(f"未归因的检出：{', '.join(rec.unattributed)}")
    return 0


def cmd_truth(args: argparse.Namespace) -> int:
    """真值。**与 `attribution` 是两个命令** —— 答案与答案的判据必须分开问。"""
    result, cached = _result(args)
    truths = list(result.run.truths)
    if args.json:
        _emit_json({"from_products": cached, "truths": [t.to_dict() for t in truths]})
        return 0
    print(f"注入的根因 {len(truths)} 个{_note(cached)}")
    print()
    for t in truths:
        print(f"[{t.root_cause_id}] {t.kind} @{t.at_step} × {t.duration_steps} 步  幅度 {fmt(t.magnitude)}")
        print(f"    what={t.what}  when={t.when_step}  resource={t.resource}  "
              f"config={t.config_key}={fmt(t.config_value)}  ckpt={t.checkpoint or '—'}")
        print(f"    recovery={t.recovery}")
        print("    chain: " + " → ".join(h.key for h in t.chain))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    result, cached = _result(args)
    ver = result.verify_result
    if args.json:
        _emit_json({"from_products": cached, **ver.to_dict()})
        return 0
    print(f"第三方校验（生成解释的人不给自己打分）{_note(cached)}")
    print(f"alpha_sigma = {fmt(ver.alpha_sigma, sig=2)}")
    print()
    print("P / F / h 探针")
    for p in ("P", "F", "h"):
        n = sum(1 for x in ver.probes if x.probe == p)
        print(_row([(p, 4), (f"{ver.probe_pass_rate(p) * 100:.0f}%", 8), (f"{n} 个", 8)]))
    print()
    print("反事实反向链（去掉某个根因，重跑整条流水线）")
    for c in ver.counterfactuals:
        print(_row([(c.root_key, 24), (f"{c.n_attributions_before}→{c.n_attributions_after}", 10),
                    (f"同类剩 {c.same_kind_after}", 14),
                    ("链消失" if c.chain_disappeared else "链还在", 10),
                    ("✓" if c.passed else "✗", 4)]))
    print()
    print("α 标定曲线（Δ/σ）")
    if ver.curves:
        fs = [f for f, _ in ver.curves[0].points]
        print(_row([("根因", 18)] + [(f"f={f:g}", 9) for f in fs] + [("f*", 8), ("单调", 6)]))
        for c in ver.curves:
            print(_row([(c.kind, 18)] + [(fmt(d, sig=3), 9) for _f, d in c.points]
                       + [(fmt(c.f_star, sig=2) if c.f_star >= 0 else "—", 8),
                          ("是" if c.monotone else "否", 6)]))
    print()
    print("对齐消融")
    for a in ver.ablations:
        print(_row([(a.label, 22), (f"{a.n_roots_recovered}/{a.n_roots_total}", 10),
                    (f"{a.recall * 100:.0f}%", 8), ((a.error[:60] if a.error else "—"), 60)]))
    print(f"对齐 lift = {ver.alignment_lift() * 100:.1f}%")
    return 0


def cmd_expectations(args: argparse.Namespace) -> int:
    result, cached = _result(args)
    rows = expectation_rows(result)
    if args.json:
        _emit_json({"from_products": cached, "rows": rows})
        return 0
    print(f"设计期望 vs 实测 —— **允许期望被推翻**{_note(cached)}")
    print()
    print(_row([("期望", 46), ("声明", 6), ("实测", 6), ("判定", 12)]))
    for r in rows:
        print(_row([(r["key"], 46), ("是" if r["declared"] else "否", 6),
                    ("成立" if r["observed"] else "不成立", 6),
                    ("✓ 兑现" if r["held"] else "✗ 被推翻", 12)]))
    print()
    refuted = [r for r in rows if not r["held"]]
    if refuted:
        print(f"被推翻 {len(refuted)} 条：")
        for r in refuted:
            print(f"  ✗ {r['key']} —— {r['note']}")
    return 0


def cmd_acceptance(args: argparse.Namespace) -> int:
    result, cached = _result(args)
    rows = acceptance_checks(result)
    adv = run_adversarial()
    n_ok = sum(1 for r in rows if r["passed"])
    n_adv = sum(1 for a in adv if a.passed)
    if args.json:
        _emit_json(
            {
                "from_products": cached,
                "checks": rows,
                "passed": n_ok,
                "total": len(rows),
                "adversarial": [a.to_dict() for a in adv],
                "adversarial_passed": n_adv,
                "adversarial_total": len(adv),
            }
        )
        return 0
    print(f"验收 {n_ok}/{len(rows)}{_note(cached)}")
    print()
    print(_row([("验收项", 32), ("实测", 14), ("门槛", 14), ("关系", 6), ("通过", 6)]))
    for r in rows:
        print(_row([(r["check"], 32), (fmt(r["value"], sig=5), 14), (fmt(r["threshold"], sig=5), 14),
                    (r["op"], 6), ("✓" if r["passed"] else "✗", 6)]))
    print()
    print(f"对抗测试 {n_adv}/{len(adv)}（**不依赖产物**：只吃配置与合成事件）")
    for a in adv:
        print(_row([("✓" if a.passed else "✗", 4), (a.name, 58), (a.detail[:70], 70)]))
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    """事件流。**读的是 `data/run/events.jsonl`，不是 `out/`。**

    这是唯一一个不看产物的读命令，因为它看的是**输入**。
    文件不在就如实说「还没采集」—— 不悄悄在内存里造一份，
    那会让人以为磁盘上有一份其实不存在的记录。
    """
    path = _data_dir(args) / "run" / "events.jsonl"
    if not path.is_file():
        msg = f"还没有采集事件流：{path}（先跑 `run`）"
        if args.json:
            _emit_json({"path": str(path), "exists": False, "n": 0, "by_source": {}, "by_kind": {}})
        else:
            print(msg)
        return 1
    events = load_events(path)
    by_source: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for e in events:
        by_source[e.source] = by_source.get(e.source, 0) + 1
        by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
    if args.source:
        events = tuple(e for e in events if e.source == args.source)
    if args.json:
        _emit_json(
            {
                "path": str(path),
                "exists": True,
                "n": len(events),
                "by_source": dict(sorted(by_source.items())),
                "by_kind": dict(sorted(by_kind.items())),
                "events": [e.to_dict() for e in events[: args.limit]] if args.limit else [],
            }
        )
        return 0
    print(f"事件流 {path} ｜ {len(events)} 个事件")
    print()
    print(_row([("来源", 12), ("事件数", 10), ("是否带 step", 12)]))
    for s, n in sorted(by_source.items()):
        carries = any(e.source == s and e.step is not None for e in load_events(path))
        print(_row([(s, 12), (str(n), 10), ("是" if carries else "否（必须靠对齐）", 12)]))
    print()
    print(_row([("事件类型", 18), ("个数", 8)]))
    for k, n in sorted(by_kind.items(), key=lambda kv: (-kv[1], kv[0])):
        print(_row([(k, 18), (str(n), 8)]))
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    result, cached = _result(args)
    lin = result.lineage
    if args.dot:
        if args.json:
            _emit_json({"from_products": cached, "dot": lin.to_dot()})
            return 0
        print(lin.to_dot(), end="")
        return 0
    if args.node:
        if not lin.has(args.node):
            print(f"没有这个节点：{args.node}", file=sys.stderr)
            return 2
        if args.json:
            _emit_json(
                {
                    "from_products": cached,
                    "node": lin.node(args.node).to_dict(),
                    "parents": list(lin.parents(args.node)),
                    "children": list(lin.children(args.node)),
                    "path_to_root": list(lin.path_to_root(args.node)),
                    "depth": lin.depth(args.node),
                }
            )
            return 0
        n = lin.node(args.node)
        print(f"{n.id}  [{n.kind}]  {n.label}")
        print(f"父节点：{', '.join(lin.parents(args.node)) or '（无）'}")
        print(f"子节点：{', '.join(lin.children(args.node)) or '（无）'}")
        print(f"到根：{' ← '.join(lin.path_to_root(args.node))}")
        return 0
    if args.json:
        _emit_json({"from_products": cached, **lin.to_dict()})
        return 0
    print(f"血缘：{len(lin.nodes)} 节点 / {len(lin.edges)} 边{_note(cached)}")
    kinds: dict[str, int] = {}
    for n in lin.nodes:
        kinds[n.kind] = kinds.get(n.kind, 0) + 1
    for k, v in sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0])):
        print(_row([(k, 14), (str(v), 6)], right=(1,)))
    print()
    print(f"根节点：{', '.join(lin.roots())}")
    print(f"叶节点 {len(lin.leaves())} 个")
    return 0


# ---------------------------------------------------------------------- 参数


#: 项目根：`src/{pkg}/cli.py` → 上三级。**所有相对路径默认值都锚到这里。**
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent


def _anchor_args(args: argparse.Namespace) -> None:
    """把 `--config` / `--out` / `--root` 的**相对路径**锚定到项目根。

    契约是 `cd <项目目录> && python main.py ...`，从项目目录里跑时锚不锚一个样
    —— 缺陷因此**长期隐形**。一旦从别处调用（例如跨项目的批量核对
    `python L5-xxx/main.py check`），`"configs/default.json"` 就指向了 cwd 下一个
    不存在的位置：严格 `load_config` 直接崩，宽松的会**静默回退 DEFAULTS**
    （更糟 —— 不报错，只是悄悄换了一套口径）。

    **在 `main()` 里紧跟 `parse_args` 锚一次**，比在每个调用点各锚一次可靠：
    漏掉一个调用点，就是一条只在某种调用方式下才现形的路径。

    绝对路径原样使用。语义与 `pytest` 的 rootdir / `cargo` / `npm --prefix` 一致。
    """
    for name in ("config", "out", "root"):
        value = getattr(args, name, None)
        if isinstance(value, str) and value:
            p = Path(value)
            if not p.is_absolute():
                setattr(args, name, str(_PKG_ROOT / p))

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="traininggraphfactory",
        description="TrainingGraphFactory：Training Runtime 的数字孪生。",
    )
    p.add_argument("--config", default=DEFAULT_CONFIG, help=f"配置文件（默认 {DEFAULT_CONFIG}）；相对路径按项目根解析")
    p.add_argument("--out", default="out", help="产物目录（默认 out）；相对路径按项目根解析")
    p.add_argument("--json", action="store_true", help="只输出 JSON")
    sub = p.add_subparsers(dest="cmd", required=True)

    # 子命令上再挂一份 `--config/--out/--json`，**默认值用 SUPPRESS**。
    # 用普通默认值的话，子解析器会把主解析器已经设好的值覆写掉：
    # `traininggraphfactory --json summary` 里的 `--json` 会被子命令的 `False` 吃掉。
    for name in READ_ONLY + WRITE:
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        sp.add_argument("--out", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        sp.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="只输出 JSON")

    sub.choices["job"].add_argument("--node", default="", help="查某个实例节点")
    sub.choices["job"].add_argument("--step", type=int, default=None, help="查某一步的 job 节点")
    sub.choices["state"].add_argument("--step", type=int, default=None, help="查某一步的状态")
    sub.choices["resource"].add_argument("--step", type=int, default=None, help="查某一步的瓶颈")
    sub.choices["resource"].add_argument("--resource", default="", help="查某个资源的压力")
    sub.choices["anomalies"].add_argument("--signal", default="", help="只看某个信号的检出")
    sub.choices["attribution"].add_argument("--kind", default="", help="只问某一种根因")
    sub.choices["lineage"].add_argument("--node", default="", help="查某个血缘节点")
    sub.choices["lineage"].add_argument("--dot", action="store_true", help="输出 DOT")
    sub.choices["events"].add_argument("--source", default="", help="只看某一路来源")
    sub.choices["events"].add_argument("--limit", type=int, default=0, help="JSON 里最多带几个事件（0 = 不带）")
    sub.choices["run"].add_argument("--no-capture", action="store_true", help="不采集事件流到 data/run/")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _anchor_args(args)
    fn = {
        "run": cmd_run,
        "visualize": cmd_visualize,
        "summary": cmd_summary,
        "graphs": cmd_graphs,
        "job": cmd_job,
        "state": cmd_state,
        "resource": cmd_resource,
        "align": cmd_align,
        "anomalies": cmd_anomalies,
        "attribution": cmd_attribution,
        "truth": cmd_truth,
        "verify": cmd_verify,
        "expectations": cmd_expectations,
        "acceptance": cmd_acceptance,
        "lineage": cmd_lineage,
        "events": cmd_events,
    }[args.cmd]
    if args.json:
        with _quiet_prose():
            return fn(args)
    return fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
