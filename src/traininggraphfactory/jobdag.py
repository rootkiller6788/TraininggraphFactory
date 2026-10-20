"""**第一张图：Job DAG。** 回答「这次运行有哪些作业、谁依赖谁」。

两层结构，缺一不可：

- **声明层**（`TEMPLATE`）：8 个节点、9 条边。它是这次运行的作业拓扑，
  与跑了多少步无关。
- **实例层**（展开后）：`train` 声明一次，运行时展开成 120 个节点。于是
  「第 55 步那个 train 节点依赖谁」是个能查的问题。

**只吃 `job` 事件。** 这不是过滤，是拒收：如果一条 `metric` 采样带着 step 号
混进来，那说明事件流坏了（全项目只有 `job` 允许带 step），`GraphError` 立刻炸，
而不是悄悄把异质事件当成作业节点画进图里。

构建必须与输入顺序无关 —— 这是「三张图可独立构建」的机器判据。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .schema import Event, JobDag, JobNode

__all__ = [
    "GraphError",
    "JOB_SOURCE",
    "TEMPLATE",
    "TEMPLATE_EDGES",
    "build_job_dag",
    "build_job_dag_from_events",
    "job_nodes_at",
]


class GraphError(RuntimeError):
    """事件流与图的声明不一致。**宁可炸，也不画一张看起来对的图。**"""


JOB_SOURCE = "job"

# 声明层的节点：`(id, label)`。对应 README 的
# `Load Model → Load Data → Preprocess → Train {checkpoint, evaluate, log} → Export`。
TEMPLATE: tuple[tuple[str, str], ...] = (
    ("load_model", "Load Model"),
    ("load_data", "Load Data"),
    ("preprocess", "Preprocess"),
    ("train", "Train"),
    ("checkpoint", "Checkpoint"),
    ("evaluate", "Evaluate"),
    ("log", "Log"),
    ("export", "Export"),
)

# 声明层的边 + **实例化规则**。规则必须写明，否则「展开」就成了一句空话：
#   `seq`        同一个模板的相邻实例串起来（`train@1 ← train@0`）
#   `step_match` 两边在同一步都有实例时相连（`checkpoint@12 ← train@12`）
#   `last_first` 左端的最后一个实例 → 右端的第一个实例
TEMPLATE_EDGES: tuple[tuple[str, str, str], ...] = (
    ("load_model", "preprocess", "last_first"),
    ("load_data", "preprocess", "last_first"),
    ("preprocess", "train", "last_first"),
    ("train", "train", "seq"),
    ("train", "checkpoint", "step_match"),
    ("train", "evaluate", "step_match"),
    ("train", "log", "last_first"),
    ("train", "export", "last_first"),
    ("checkpoint", "export", "last_first"),
)


def build_job_dag(run: Any) -> JobDag:
    return build_job_dag_from_events(run.events, n_steps=len(run.traces))


def build_job_dag_from_events(events: Iterable[Event], n_steps: int) -> JobDag:
    """从**整条**事件流建 Job DAG —— 但异质事件一律拒收。

    拒收规则只有一条，却是本模块的立身之本：**只有 `job` 允许带 step 号**。
    所以「非 job 事件带 step」= 事件流坏了。至于「非 job 事件不带 step」，
    那是正常的，直接无视。
    """
    evs = sorted(events, key=lambda e: (e.t_local, e.id))
    foreign = [e for e in evs if e.source != JOB_SOURCE and e.step is not None]
    if foreign:
        ex = foreign[0]
        raise GraphError(
            f"{len(foreign)} 条非 job 事件带了 step 号（例如 {ex.id} @ {ex.source}）—— "
            "全项目只有 job 允许带 step，事件流是坏的，不能拿来建 Job DAG。"
        )
    job = [e for e in evs if e.source == JOB_SOURCE]
    if not job:
        raise GraphError("事件流里一条 job 事件都没有 —— 没有作业，谈不上 DAG。")

    steps = sorted({e.step for e in job if e.step is not None})
    if not steps:
        raise GraphError("job 事件全都没有 step 号 —— Job DAG 无锚可依。")
    last = max(steps)

    counts: dict[tuple[str, int], int] = {}

    def bump(template: str, step: int) -> None:
        counts[(template, step)] = counts.get((template, step), 0) + 1

    for e in job:
        s = e.step
        if s is None:
            continue
        if e.kind in ("step_begin", "step_end", "config_change"):
            bump("train", s)
        elif e.kind in ("ckpt_begin", "ckpt_end"):
            bump("checkpoint", s)
        elif e.kind in ("eval_begin", "eval_end"):
            bump("evaluate", s)
        elif e.kind == "state_change":
            # 状态转移属于它发生的那一步。`INIT→PREPARING` 是 preprocess 的活，
            # 其余（含 `PREPARING→TRAINING`）都记在 train 上 —— 于是 preprocess
            # 只在第 0 步有一个实例，而 train 从第 0 步起每一步都有一个。
            to = str(e.payload.get("to", ""))
            bump("preprocess" if to == "PREPARING" else "train", s)
        elif e.kind == "run_begin":
            bump("load_model", s)
        elif e.kind == "run_end":
            bump("export", s)

    labels = dict(TEMPLATE)

    def node_id(template: str, step: int) -> str:
        return f"{template}@{step}"

    # 实例：模板 → 它出现的步（升序）
    inst: dict[str, list[int]] = {t: [] for t, _ in TEMPLATE}
    for (t, s), _n in sorted(counts.items()):
        inst.setdefault(t, []).append(s)
    for t in inst:
        inst[t] = sorted(set(inst[t]))

    # 三个只有一次、没有步锚点的作业：锚在它能确定的位置上
    if not inst["load_model"]:
        inst["load_model"] = [0]
    if not inst["load_data"]:
        inst["load_data"] = [0]
    if not inst["preprocess"]:
        inst["preprocess"] = [0]
    # `log` 是 metric 的汇：它整条运行都在记，锚在第 0 步
    inst["log"] = [0]
    if not inst["export"]:
        inst["export"] = [last]

    nodes: list[JobNode] = []
    for t, _label in TEMPLATE:
        for s in inst[t]:
            nodes.append(
                JobNode(
                    id=node_id(t, s),
                    template=t,
                    label=labels[t],
                    step_lo=s,
                    step_hi=s,
                    n_events=counts.get((t, s), 0),
                )
            )
    node_ids = {n.id for n in nodes}

    edges: list[tuple[str, str]] = []
    for a, b, rule in TEMPLATE_EDGES:
        if rule == "seq":
            for x, y in zip(inst[a], inst[a][1:]):
                edges.append((node_id(a, x), node_id(a, y)))
        elif rule == "step_match":
            for s in sorted(set(inst[a]) & set(inst[b])):
                edges.append((node_id(a, s), node_id(b, s)))
        elif rule == "last_first":
            if inst[a] and inst[b]:
                edges.append((node_id(a, inst[a][-1]), node_id(b, inst[b][0])))
        else:
            raise GraphError(f"未声明的实例化规则：{rule!r}")

    edges = [(a, b) for a, b in edges if a in node_ids and b in node_ids]
    edges = sorted(set(edges))

    # 多源点不修：`load_model` 与 `load_data` 本来就是两项可以并行的独立作业，
    # 一个 DAG 有多个源点是正常的。`topo_order` 用「入度 0 的节点全进就绪队列」
    # 处理它，不需要人为补边。
    return JobDag(
        nodes=tuple(sorted(nodes, key=lambda n: n.id)),
        edges=tuple(edges),
        template_nodes=TEMPLATE,
        template_edges=tuple((a, b) for a, b, _r in TEMPLATE_EDGES),
        n_steps=n_steps,
    )


def job_nodes_at(dag: JobDag, step: int) -> tuple[JobNode, ...]:
    """某一步上活跃的作业节点（可能有多个：train 与 checkpoint 同时在）。"""
    return tuple(n for n in dag.nodes if n.step_lo <= step <= n.step_hi)
