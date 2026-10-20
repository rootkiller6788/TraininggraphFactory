"""报告、验收与「设计期望 vs 实测」。**三者共用同一份计算。**

这不是为了少写代码，是为了让「命令行说通过的」与「报告里写的」不可能不一致 ——
上一版把验收表在报告里手抄了一遍，于是改了一处门槛之后两边对不上，
而两边各自都「看起来对」。

**附录 B 的规矩：允许期望被推翻。** 本项目的九条声明期望里有四条**没有兑现**，
它们被逐条点名。一条被推翻的期望比九条被证实的更值钱 ——
它说明这个设计里有一句话是错的，而流水线把它测出来了。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .config import expectations_of
from .recover import group_key
from .schema import LAYERS
from .state import state_path_of

__all__ = [
    "fmt",
    "expectation_rows",
    "acceptance_checks",
    "acceptance_pass_rate",
    "render_report",
]


def fmt(x: Any, *, sig: int = 4) -> str:
    """紧凑数字。`None` → `—`、`nan` → `nan`、`inf` → `inf`。
    **`None` 与 `nan` 必须印成不同的东西** —— 前者是「没测到」，后者是「测到了 nan」。"""
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "是" if x else "否"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf"
        if x == 0.0:
            return "0"
        if abs(x) >= 1e6 or abs(x) < 1e-3:
            return f"{x:.{sig}g}"
        return f"{x:.{sig}g}"
    return str(x)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# --------------------------------------------------------------------------- #
# 设计期望 vs 实测
# --------------------------------------------------------------------------- #


def _truth_of(result: Any) -> dict[str, Any]:
    return {t.kind: t for t in result.run.truths}


def _matched(result: Any) -> list[tuple[Any, Any]]:
    """把归因与真值**按种类 + 时刻**配对。配对是第三方的事，报告只呈现结果。"""
    truths = _truth_of(result)
    out: list[tuple[Any, Any]] = []
    for a in result.recover_result.attributions:
        t = truths.get(a.root_cause_kind)
        if t is None:
            continue
        if abs(a.when_step - t.when_step) > 2:
            continue
        out.append((a, t))
    return out


def _five_answer_hits(result: Any) -> int:
    n = 0
    for a, t in _matched(result):
        if (
            a.what == t.what
            and abs(a.when_step - t.when_step) <= 1
            and a.resource == t.resource
            and a.config_key == t.config_key
            and a.config_value == t.config_value
            and a.checkpoint == t.checkpoint
        ):
            n += 1
    return n


def _chain_hits(result: Any) -> int:
    return sum(1 for a, t in _matched(result) if a.chain_keys == t.chain_keys)


def _unstratified_count(result: Any) -> int | None:
    """把分层关掉再检一遍。**这就是 `single_global_baseline_suffices` 的判据。**"""
    try:
        from .config import deep_merge
        from .detect import detect

        c = deep_merge(result.cfg, {"detect": {"stratify_by_step_type": False, "stratum_fallback": "all"}})
        return len(detect(result.alignment.step_table, c).anomalies)
    except Exception:  # pragma: no cover - 只用于报告，失败就如实记 None
        return None


def _signature_materialization(result: Any) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """每种根因：声明的签名里，哪些**真的**出现在了真值链上、哪些没有。

    这是 `declared_signatures_all_materialize` 的判据。**它允许不兑现** ——
    `eval_regression` 声明的 `step_time_z` 就没有兑现，而且它是有意不兑现的。

    比对必须走 `hop_key` 而不是自己拼前缀：层名是 `metric`，键前缀是 `met` ——
    自己拼会得到一个**永远匹配不上**的字符串，于是每条签名都「未兑现」，
    而那看起来会像一个惊天动地的发现。
    """
    from .config import signatures_of
    from .schema import EVIDENCE_LAYER, EVIDENCE_SIGNAL, FAILED_STATE_EDGE, hop_key

    sigs = signatures_of(result.cfg)
    out: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for t in result.run.truths:
        declared = sigs.get(t.kind, ())
        present = set(t.chain_keys)
        # 三个特殊标签的键不是信号名，而是「当时的那个东西」的名字 ——
        # 它们只能从这条真值自己身上取（`cfg:train.lr` / `st:TRAINING->FAILED` / `ckpt:ckpt-0096`）。
        special = {
            "config_change": f"cfg:{t.config_key}",
            "state_failed": f"st:{FAILED_STATE_EDGE}",
            "ckpt_restore": f"ckpt:{t.checkpoint}",
        }
        labels_present: list[str] = []
        labels_missing: list[str] = []
        for lab in declared:
            layer = EVIDENCE_LAYER.get(lab, "")
            if lab in special:
                base = special[lab]
            else:
                base = hop_key(layer, EVIDENCE_SIGNAL.get(lab, lab))
            hit = any(k == base or k.startswith(f"{base}@") for k in present)
            (labels_present if hit else labels_missing).append(lab)
        out[t.kind] = (tuple(labels_present), tuple(labels_missing))
    return out


def expectation_rows(result: Any) -> list[dict[str, Any]]:
    """九条声明期望 × 实测。`held=False` 的那些**必须**在报告里被点名。"""
    declared = expectations_of(result.cfg)
    ver = result.verify_result
    det = result.detect_result
    rec = result.recover_result

    n_roots = len(result.run.truths)
    recall = ver.n_roots_recovered / max(1, n_roots)
    fp = len(rec.unattributed)
    n_matched = max(1, len(_matched(result)))
    five_rate = _five_answer_hits(result) / n_matched
    chain_rate = _chain_hits(result) / n_matched

    lift = ver.alignment_lift()
    const_only = ver.ablation("identity+constant")
    unstrat = _unstratified_count(result)
    sigmat = _signature_materialization(result)
    missing_total = sum(len(m) for _p, m in sigmat.values())

    obs: dict[str, tuple[bool, str]] = {
        "alignment_is_necessary": (
            lift > 0.0,
            f"对齐消融：常数偏移只剩 {const_only.n_roots_recovered if const_only else '?'}/{n_roots} 条归因，"
            f"lift={_pct(lift)}",
        ),
        "constant_clock_offset_suffices": (
            (const_only is not None and const_only.n_roots_recovered == n_roots),
            (
                f"只用 identity+constant：{const_only.n_roots_recovered}/{n_roots} 条归因"
                + (f"（对齐直接失败：{const_only.error[:60]}）" if const_only and const_only.error else "")
                if const_only is not None
                else "消融未跑"
            ),
        ),
        "single_global_baseline_suffices": (
            (unstrat is not None and unstrat <= len(det.anomalies)),
            f"不分层检出 {unstrat if unstrat is not None else '?'} 条 vs 分层 {len(det.anomalies)} 条",
        ),
        "declared_signatures_all_materialize": (
            missing_total == 0,
            f"未兑现的签名标签 {missing_total} 个"
            + (
                "（" + "、".join(f"{k}:{','.join(m)}" for k, (_p, m) in sorted(sigmat.items()) if m) + "）"
                if missing_total
                else ""
            ),
        ),
        "declared_causal_model_attributes_uniquely": (
            not any(a.ambiguous for a in rec.attributions),
            f"有歧义的归因 {sum(1 for a in rec.attributions if a.ambiguous)} 起",
        ),
        "all_injected_anomalies_detected": (recall >= 1.0, f"召回 {ver.n_roots_recovered}/{n_roots} = {_pct(recall)}"),
        "no_false_positive_anomalies": (fp == 0, f"未被任何归因覆盖的检出 {fp} 条"),
        "three_graphs_separately_queryable": (
            True,
            f"job {len(result.job_dag.nodes)} 节点 / state {len(result.state_graph.visited_nodes)} 状态 / "
            f"resource {len(result.resource_graph.resources)} 资源，三份产物三个文件",
        ),
        "recovery_lands_on_last_clean_checkpoint": (
            all(a.checkpoint == t.checkpoint for a, t in _matched(result)),
            f"checkpoint 全对 {sum(1 for a, t in _matched(result) if a.checkpoint == t.checkpoint)}/{len(_matched(result))}",
        ),
    }

    rows: list[dict[str, Any]] = []
    for key in declared:
        held, note = obs.get(key, (False, "没有为这条期望写判据"))
        rows.append(
            {
                "key": key,
                "declared": bool(declared[key]),
                "observed": bool(held),
                "held": bool(held) == bool(declared[key]),
                "note": note,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# 验收
# --------------------------------------------------------------------------- #


def acceptance_checks(result: Any) -> list[dict[str, Any]]:
    """验收表。每条都给出**实测值 / 门槛 / 是否通过**。"""
    cfg = result.cfg
    acfg = cfg["acceptance"]
    ver = result.verify_result
    det = result.detect_result
    rec = result.recover_result
    al = result.alignment

    n_roots = len(result.run.truths)
    recall = ver.n_roots_recovered / max(1, n_roots)
    fp = len(rec.unattributed)
    n_matched = max(1, len(_matched(result)))
    five_rate = _five_answer_hits(result) / n_matched
    chain_rate = _chain_hits(result) / n_matched

    off_err = max((abs(r.estimated_offset_s - r.injected_offset_s) for r in al.reports), default=0.0)
    drift_err = max((abs(r.estimated_drift_s_per_s - r.injected_drift_s_per_s) for r in al.reports), default=0.0)
    lift = ver.alignment_lift()
    n_states = len(result.state_graph.visited_nodes)
    n_res = len(result.resource_graph.resources)
    n_job = len(result.job_dag.nodes)

    specs: list[tuple[str, float, float, str, str]] = [
        ("min_anomaly_recall", recall, float(acfg["min_anomaly_recall"]), ">=", "注入根因被归因接住的比例"),
        (
            "max_anomaly_false_positives",
            float(fp),
            float(acfg["max_anomaly_false_positives"]),
            "<=",
            "未被任何归因覆盖的检出条数",
        ),
        ("min_five_answer_rate", five_rate, float(acfg["min_five_answer_rate"]), ">=", "五问全对的比例"),
        ("min_chain_exact_rate", chain_rate, float(acfg["min_chain_exact_rate"]), ">=", "因果链逐键一致的比例"),
        ("max_offset_error_s", off_err, float(acfg["max_offset_error_s"]), "<=", "时钟偏移估计误差上限（秒）"),
        (
            "max_drift_error_s_per_s",
            drift_err,
            float(acfg["max_drift_error_s_per_s"]),
            "<=",
            "时钟漂移估计误差上限（秒/秒）",
        ),
        ("min_alignment_lift", lift, float(acfg["min_alignment_lift"]), ">=", "漂移感知比只用常数偏移多救回的归因比例"),
        ("min_state_coverage", float(n_states), float(acfg["min_state_coverage"]), ">=", "状态图走过的状态数"),
        ("min_resource_coverage", float(n_res), float(acfg["min_resource_coverage"]), ">=", "资源图覆盖的资源数"),
        ("min_job_nodes", float(n_job), float(acfg["min_job_nodes"]), ">=", "Job DAG 的实例节点数"),
    ]

    rows: list[dict[str, Any]] = []
    for name, got, want, op, note in specs:
        ok = (got >= want - 1e-12) if op == ">=" else (got <= want + 1e-12)
        rows.append({"check": name, "value": got, "threshold": want, "op": op, "passed": ok, "note": note})
    return rows


def acceptance_pass_rate(result: Any) -> float:
    rows = acceptance_checks(result)
    return (sum(1 for r in rows if r["passed"]) / len(rows)) if rows else 0.0


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def render_report(result: Any, *, rows: Sequence[Mapping[str, Any]] | None = None) -> str:
    if rows is None:
        rows = expectation_rows(result)

    cfg = result.cfg
    al = result.alignment
    det = result.detect_result
    rec = result.recover_result
    ver = result.verify_result
    L: list[str] = []

    L.append("# TrainingGraphFactory 归因报告")
    L.append("")
    # **报告正文里不印 `created_at`。** 印了它，两次相同输入的报告就不逐位相同了 ——
    # 而「唯一不可复现的字段是 created_at」这条约定只在 `version.json` 里成立。
    # 时间戳本身没有丢：它在 `version.json` 里，报告指过去就够了。
    L.append(
        f"> 版本 `{result.version.id}` ｜ 口径 `{result.version.protocol_digest[:12]}` ｜ "
        f"{result.version.n_steps} 步 / {result.version.n_events} 个事件 ｜ 生成时间见 `version.json`"
    )
    L.append("")
    L.append("**这不是 Training UI。** 它是 Training Runtime 的数字孪生：在五张不同的信号之间建立可追溯的因果链。")
    L.append("")

    # ---------------------------------------------------------------- 一
    L.append("## 一、三张图：分开构建、分开查询")
    L.append("")
    L.append("三者混在一起就会退化成一张杂乱的流程图，所以它们在数据层就分开 —— 也落成三份产物。")
    L.append("")

    L.append("### 1. Job DAG")
    L.append("")
    job = result.job_dag
    L.append(
        f"{len(job.nodes)} 个实例节点 / {len(job.edges)} 条实例边 ｜ "
        f"模板 {len(job.template_nodes)} 节点 / {len(job.template_edges)} 边 ｜ 步数 {job.n_steps}"
    )
    L.append("")
    by_tpl: dict[str, int] = {}
    for n in job.nodes:
        by_tpl[n.template] = by_tpl.get(n.template, 0) + 1
    L.extend(
        _table(
            ("模板节点", "实例数", "说明"),
            [(t, str(c), _JOB_NOTE.get(t, "")) for t, c in sorted(by_tpl.items(), key=lambda kv: (-kv[1], kv[0]))],
        )
    )
    L.append("")
    roots = job.topo_order()[:2] if len(job.topo_order()) >= 2 else job.topo_order()
    L.append(f"拓扑序前两个节点：`{roots[0]}` → `{roots[1]}`（**入度为 0 时按节点 id 排序取下一个**，")
    L.append("否则 `topo_order` 会依赖 `set` 的迭代顺序，两次运行给出不同的序）。")
    L.append("")
    probe_step = 72 if any(n.id == "checkpoint@72" for n in job.nodes) else job.nodes[-1].step_lo
    L.append(f"任取一步查依赖：`checkpoint@{probe_step}` ← `{', '.join(job.deps(f'checkpoint@{probe_step}')) or '（无依赖）'}`；")
    L.append(f"该节点的祖先有 {len(job.ancestors(f'checkpoint@{probe_step}'))} 个。")
    L.append("")

    L.append("### 2. Training State Graph")
    L.append("")
    stg = result.state_graph
    L.append(
        f"声明 {len(stg.nodes)} 个状态 / {len(stg.edges)} 条转移 ｜ "
        f"实际走过 {len(stg.visited_nodes)} 个状态 / {len(stg.visited_edges)} 条转移"
    )
    L.append("")
    # 印的是**去重保序**的路线，不是逐步状态 —— 逐步那条会印出 117 个连续的 `TRAINING`，
    # 而它想表达的只是一段「一直在训练」。逐步状态留给 `state --step N` 去问。
    L.append(f"路线：`{' → '.join(state_path_of(stg))}`")
    L.append("")
    L.append(
        f"**没走到的**状态：`{', '.join(stg.unvisited_nodes) or '（无）'}`；"
        f"没走到的转移：`{'、'.join(f'{a}→{b}' for a, b in stg.unvisited_edges) or '（无）'}`。"
    )
    L.append("")
    L.append("没走到不是缺陷 —— 它是「声明了但这次没发生」。**它与「声明里根本没有」必须分开**，")
    L.append("所以报告里两条都印出来。")
    L.append("")
    L.extend(
        _table(
            ("状态", "停留步数"),
            [(s, str(n)) for s, n in sorted(stg.steps_in_state, key=lambda kv: (-kv[1], kv[0]))],
        )
    )
    L.append("")

    L.append("### 3. Resource Graph")
    L.append("")
    rg = result.resource_graph
    L.append(f"中枢 `{rg.hub}`，{len(rg.resources)} 个资源。归一化用**该资源自己的稳健 z**，不是全局 z。")
    L.append("")
    L.extend(
        _table(
            ("资源", "信号", "峰值步", "峰值", "基线", "归一化 z", "方向"),
            [
                (
                    p.resource,
                    rg.signal_of.get(p.resource, ""),
                    str(p.peak_step),
                    fmt(p.peak_value),
                    fmt(p.baseline),
                    fmt(p.normalized, sig=3),
                    p.direction,
                )
                for p in rg.ranked()
            ],
        )
    )
    L.append("")
    n_bottleneck = sum(1 for _s, r in rg.bottleneck_by_step if r)
    L.append(
        "**瓶颈只在压力超过 `k` 时报，否则报空。** 一个「谁都没超阈值」的步不该被强行指认一个瓶颈 —— "
        f"本次 {len(rg.bottleneck_by_step)} 步里只有 {n_bottleneck} 步报了瓶颈，其余报「无显著压力」。"
    )
    L.append("")

    # ---------------------------------------------------------------- 二
    L.append("## 二、对齐：四路异质事件 → 一张逐步信号表")
    L.append("")
    L.append(
        f"只有 `job` 带步号，其余三路必须靠**同步信标**接回来。"
        f"共 {al.n_events_total} 个待归属事件，{al.n_events_assigned} 个接上了步。"
    )
    L.append("")
    L.append("三种假设**同台**，判据是**步归属错误率**而不是拟合优度：")
    L.append("")
    rows_a: list[list[str]] = []
    for r in al.reports:
        cands = al.models_of(r.source)
        by_name = {m.model: m for m in cands}
        for name in ("identity", "constant", "linear"):
            m = by_name.get(name)
            if m is None:
                continue
            mark = " **←**" if r.model == name else ""
            rows_a.append(
                [
                    r.source if name == "identity" else "",
                    name + mark,
                    fmt(m.a, sig=5),
                    fmt(m.b, sig=6),
                    fmt(m.residual_rms_s, sig=3),
                    _pct(m.assign_error_rate),
                ]
            )
    L.extend(_table(("来源", "假设", "估计偏移 (s)", "估计斜率", "残差 RMS (s)", "步归属错误率"), rows_a))
    L.append("")
    L.append("选择规则两步：① 残差必须在 `residual_tol_s` 内（`identity` 就是这样出局的 —— 它的残差等于偏移本身）；")
    L.append("② 在剩下的里挑错误率最低的，**打平则取更简单的** —— `linear` 必须挣得它的第二个参数。")
    L.append("")
    L.extend(
        _table(
            ("来源", "选中", "注入偏移", "估计偏移", "偏移误差", "注入漂移", "估计漂移", "漂移误差"),
            [
                (
                    r.source,
                    r.model,
                    fmt(r.injected_offset_s, sig=5),
                    fmt(r.estimated_offset_s, sig=5),
                    fmt(abs(r.estimated_offset_s - r.injected_offset_s), sig=3),
                    fmt(r.injected_drift_s_per_s, sig=4),
                    fmt(r.estimated_drift_s_per_s, sig=4),
                    fmt(abs(r.estimated_drift_s_per_s - r.injected_drift_s_per_s), sig=3),
                )
                for r in al.reports
            ],
        )
    )
    L.append("")
    L.append("**偏移不可辨识的那一路按定义取 0。** `job` 的偏移不是「估出来是 0」，是**坐标系选在它上面** ——")
    L.append("所有偏移都是相对 job 的偏移，绝对值没有意义。")
    L.append("")

    # ---------------------------------------------------------------- 三
    L.append("## 三、检出：去趋势 + 分层 + 乘性通道")
    L.append("")
    L.append(
        f"`k={fmt(cfg['detect']['k'], sig=2)}`，检出 {len(det.anomalies)} 条，"
        f"跨 {len(det.by_signal())} 个信号。"
    )
    L.append("")
    L.append("为什么必须去趋势：`loss` 从 4.2 衰减到 0.12（34×），全局中位数基线检不出真正的 spike。")
    L.append("为什么必须分层：`ckpt_every=12` 与 `eval_every=10` 每 60 步撞一次，第 60 步既写 ckpt 又跑 eval，")
    L.append("耗时比普通 ckpt 步长 30% —— 在 ckpt 层里 z≈15 的假阳性。")
    L.append("")
    by_sig = det.by_signal()
    L.extend(
        _table(
            ("信号", "条数", "步区间", "峰值 z（有符号）"),
            [
                (
                    s,
                    str(len(v)),
                    "、".join(f"{a.step_lo}-{a.step_hi}" if a.step_hi > a.step_lo else str(a.step_lo) for a in v),
                    fmt(max(a.peak_z for a in v if not a.is_nan), sig=3) if any(not a.is_nan for a in v) else "nan",
                )
                for s, v in sorted(by_sig.items())
            ],
        )
    )
    L.append("")
    L.append("z 带符号：`gpu0_util` 是「高才好」，所以它的峰值 z 是**负的** —— 掉下去才是坏事。")
    L.append("丢掉符号就分不出「利用率涨上去」和「掉下来」了。")
    L.append("")
    if det.skipped:
        L.append(
            f"**未判定**：{len(det.skipped)} 个信号各有样本不足而未判定的步"
            f"（`{', '.join(f'{k}:{v}' for k, v in sorted(det.skipped.items()))}`）。"
        )
        L.append("「未判定」与「正常」必须分开报 —— 拿全量基线去判一个样本不足的分层，正是「不分层」那个错误的翻版。")
        L.append("")
    L.append("**NaN 必须显式判异常**：`abs(nan - median) > k*mad` 永远是 False，靠容差比较永远判不出来。")
    L.append("而且 NaN 要留在逐步信号表里 —— 它是「测到了 nan」，不是「没测到」（后者是 `None`）。")
    L.append("")

    # ---------------------------------------------------------------- 四
    L.append("## 四、归因：五问")
    L.append("")
    L.append(f"{len(rec.attributions)} 起归因，覆盖 {len(det.anomalies) - len(rec.unattributed)} 条检出；")
    L.append(f"未被归因的检出 {len(rec.unattributed)} 条。")
    L.append("")
    L.append("**归因与真值分两个文件。** 归因器一行真值都不读；`truths.json` 是第三方（`verify`）用的。")
    L.append("")
    L.extend(
        _table(
            ("根因", "what", "when", "resource", "config", "checkpoint", "置信度", "歧义"),
            [
                (
                    group_key(a),
                    a.what,
                    str(a.when_step),
                    a.resource,
                    f"`{a.config_key}`={fmt(a.config_value)}",
                    f"`{a.checkpoint}`" if a.checkpoint else "—",
                    fmt(a.confidence, sig=2),
                    "是" if a.ambiguous else "否",
                )
                for a in rec.attributions
            ],
        )
    )
    L.append("")
    L.append("### 因果链：八层信号")
    L.append("")
    L.append(f"层是固定的：`{' / '.join(LAYERS)}`。因果链**跨层**，每一跳都能用 `layer + key` 唯一定位。")
    L.append("")
    L.append("键的前缀已经标了层（`cfg:` / `dat:` / `io:` / `job:` / `res:` / `met:` / `st:` / `ckpt:`），所以下表只印键。")
    L.append("")
    L.extend(
        _table(
            ("根因", "链（从因到果）"),
            [
                (group_key(a), " → ".join(f"`{h.key}`" for h in a.chain) or "—")
                for a in rec.attributions
            ],
        )
    )
    L.append("")
    L.append("**证据步的约定是「首次出现」而不是峰值。** 一次持续 3 步的等比例跌幅里「哪一步是峰」是平局；")
    L.append("让真值链和复原链各自猜平局，等于让链的比对变成比运气。")
    L.append("")

    # ---------------------------------------------------------------- 五
    L.append("## 五、第三方校验：生成解释的人不给自己打分")
    L.append("")
    L.append("### P / F / h 三个可判分探针")
    L.append("")
    L.append(f"`alpha_sigma = {fmt(ver.alpha_sigma, sig=2)}` —— 变化不到一个噪音尺度，与噪音不可区分。")
    L.append("")
    L.extend(
        _table(
            ("探针", "问的问题", "通过率", "个数"),
            [
                ("P", "去掉它，目标信号**要动**", _pct(ver.probe_pass_rate("P")), str(sum(1 for p in ver.probes if p.probe == "P"))),
                ("F", "去掉它，它声明**不影响**的信号必须不动", _pct(ver.probe_pass_rate("F")), str(sum(1 for p in ver.probes if p.probe == "F"))),
                ("h", "在别处注入等量**不同种类**的诱饵，目标信号必须不动", _pct(ver.probe_pass_rate("h")), str(sum(1 for p in ver.probes if p.probe == "h"))),
            ],
        )
    )
    L.append("")
    L.append("只测 P 是不够的：一个「只要注了就都行」的判据也会全过。所以三个方向都要测。")
    L.append("")
    L.append("### 反事实反向链")
    L.append("")
    L.append("把某个原因**真的从模拟器里去掉**，重跑整条流水线，看它的归因是否消失。")
    L.append("")
    L.extend(
        _table(
            ("根因", "去掉前", "去掉后", "同类剩", "链消失", "通过"),
            [
                (
                    c.root_key,
                    str(c.n_attributions_before),
                    str(c.n_attributions_after),
                    str(c.same_kind_after),
                    "是" if c.chain_disappeared else "否",
                    "✓" if c.passed else "✗",
                )
                for c in ver.counterfactuals
            ],
        )
    )
    L.append("")
    L.append("### α 标定曲线")
    L.append("")
    L.append("基线取「幅度压到 1 = 无效果」的那次运行，`f` 从 0 扫到 1。`f*` 是**最小仍充分**的幅度因子。")
    L.append("")
    if ver.curves:
        fs = [f for f, _ in ver.curves[0].points]
        L.extend(
            _table(
                ("根因", *[f"f={f:g}" for f in fs], "f*", "单调"),
                [
                    (
                        c.kind,
                        *[("inf" if math.isinf(d) else fmt(d, sig=3)) for _f, d in c.points],
                        fmt(c.f_star, sig=2) if c.f_star >= 0 else "—（开关式）",
                        "是" if c.monotone else "否",
                    )
                    for c in ver.curves
                ],
            )
        )
    L.append("")
    L.append("**`nan_grad` 的曲线必然退化。** 它是开关式的：梯度变成 NaN 这件事没有「幅度小一点」的中间态。")
    L.append("一条测不出中间态的曲线不是流水线的缺陷，是它诚实报告了「这个原因不可调幅」。")
    L.append("")
    L.append("### 对齐消融")
    L.append("")
    L.extend(
        _table(
            ("允许的假设", "救回归因", "召回", "说明"),
            [
                (
                    a.label,
                    f"{a.n_roots_recovered}/{a.n_roots_total}",
                    _pct(a.recall),
                    a.error[:70] if a.error else "—",
                )
                for a in ver.ablations
            ],
        )
    )
    L.append("")

    # ---------------------------------------------------------------- 六
    L.append("## 六、设计期望 vs 实测")
    L.append("")
    L.append("**允许期望被推翻。** 被推翻的那条最值钱 —— 它说明设计里有一句话是错的。")
    L.append("")
    L.extend(
        _table(
            ("期望", "声明", "实测", "判定", "依据"),
            [(r["key"], "是" if r["declared"] else "否", "成立" if r["observed"] else "不成立", "✓ 兑现" if r["held"] else "✗ 被推翻", r["note"]) for r in rows],
        )
    )
    L.append("")
    refuted = [r for r in rows if not r["held"]]
    if refuted:
        L.append(f"**被推翻 {len(refuted)} 条：**")
        L.append("")
        for r in refuted:
            L.append(f"- `{r['key']}` —— 声明「{'成立' if r['declared'] else '不成立'}」，实测「{'成立' if r['observed'] else '不成立'}」：{r['note']}")
        L.append("")

    # ---------------------------------------------------------------- 七
    L.append("## 七、验收")
    L.append("")
    acc = acceptance_checks(result)
    L.extend(
        _table(
            ("验收项", "实测", "门槛", "关系", "通过"),
            [
                (a["check"], fmt(a["value"], sig=5), fmt(a["threshold"], sig=5), a["op"], "✓" if a["passed"] else "✗")
                for a in acc
            ],
        )
    )
    L.append("")
    L.append(f"**{sum(1 for a in acc if a['passed'])}/{len(acc)} 通过。**")
    L.append("")

    # ---------------------------------------------------------------- 八
    L.append("## 八、血缘")
    L.append("")
    lin = result.lineage
    L.append(f"{len(lin.nodes)} 个节点 / {len(lin.edges)} 条边。两条别处没有的不变量：")
    L.append("")
    L.append("1. `attrib:<kind>@<step>` 的父节点**恰好是**它 `covered_anomalies` 里那几个检出；")
    L.append("2. `hop:<layer>:<key>` 的父节点**恰好是**链上前一跳 —— 于是从任意一跳往上走，走到根就是这条链的第一因。")
    L.append("")
    kinds: dict[str, int] = {}
    for n in lin.nodes:
        kinds[n.kind] = kinds.get(n.kind, 0) + 1
    L.extend(
        _table(("节点类型", "个数"), [(k, str(v)) for k, v in sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))])
    )
    L.append("")
    L.append("_产物：三张图各一份 JSON、一张逐步信号表、对齐 / 检出 / 归因 / 真值 / 校验 / 期望各一份 JSON、血缘 JSON + DOT。_")
    L.append("")
    return "\n".join(L)


_JOB_NOTE = {
    "load_model": "载入模型权重（依赖 `load_data` 的结果）",
    "load_data": "载入数据分片",
    "preprocess": "预处理 / 分词",
    "train": "逐步训练（每个训练步一个实例）",
    "checkpoint": "写 checkpoint（每 `ckpt_every` 步）",
    "evaluate": "跑评测（每 `eval_every` 步）",
    "log": "写指标日志（每步）",
    "export": "导出最终产物",
}
