"""**第二张图：Training State Graph。**

运行时经历了哪些状态、怎么迁移的。与 Job DAG 的区别很重要：Job DAG 是**声明**的
作业拓扑（跑 10 步还是 120 步都一样），状态图是**这次运行真的走过的路径**。

两条纪律：

1. **未声明的转移一律拒收。** 事件流里出现 `TRAINING -> DONE`（本项目没有这条边）
   就是事件流坏了，`GraphError` 立刻炸。状态图的价值全在「它只画真发生过的边」——
   一条悄悄补上的边会让整张图变成猜测。
2. **声明过但没走到的状态要如实报出来。** 本项目声明了 9 个状态，本次运行没走到
   `PAUSED`。把它画进图里当成走过，等于谎报覆盖；报成 `unvisited_nodes` 才是有信息量的。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .events import STATE_EDGES
from .jobdag import GraphError
from .schema import STATES, StateGraph, StateTransition

__all__ = ["build_state_graph", "DECLARED_EDGES", "state_path_of"]

DECLARED_EDGES: tuple[tuple[str, str], ...] = STATE_EDGES


def build_state_graph(
    run: Any,
    *,
    states: Sequence[str] = STATES,
    edges: Sequence[tuple[str, str]] = DECLARED_EDGES,
) -> StateGraph:
    """从状态转移事件重放出一张状态图。

    重放时会做**三重核对**，任何一条不成立就报错：

    1. 转移的起点必须是**当前状态**（不能从别处跳过来）；
    2. 这条边必须是声明过的；
    3. 两端的名字都必须是声明过的状态。
    """
    declared = tuple(edges)
    declared_set = set(declared)
    known = set(states)

    trans = tuple(sorted(run.transitions, key=lambda t: t.step))
    if not trans:
        raise GraphError("事件流里一条状态转移都没有 —— 没有路径，谈不上状态图。")

    cur = states[0]
    for t in trans:
        if t.frm not in known or t.to not in known:
            raise GraphError(f"状态转移引用了未声明的状态：{t.frm} -> {t.to}")
        if t.frm != cur:
            raise GraphError(
                f"第 {t.step} 步声称从 {t.frm} 转到 {t.to}，但当前状态是 {cur} —— "
                "状态路径断了，这张图不能信。"
            )
        if (t.frm, t.to) not in declared_set:
            raise GraphError(
                f"第 {t.step} 步走了一条未声明的转移：{t.frm} -> {t.to}"
                f"（声明的边里没有它）。"
            )
        cur = t.to

    # 每一步**结束之后**处于什么状态。没发生转移的步沿用上一步的状态。
    by_step: dict[int, list[StateTransition]] = {}
    for t in trans:
        by_step.setdefault(t.step, []).append(t)
    all_steps = sorted(by_step)
    span = list(range(min(all_steps), max(all_steps) + 1))

    path: list[tuple[int, str]] = []
    cursor = states[0]
    for s in span:
        for t in by_step.get(s, ()):
            cursor = t.to
        path.append((s, cursor))

    visited_nodes = tuple(sorted({states[0], *(t.to for t in trans)}, key=lambda x: list(states).index(x)))
    visited_edges = tuple(sorted(set((t.frm, t.to) for t in trans)))

    counts: dict[str, int] = {}
    for _s, name in path:
        counts[name] = counts.get(name, 0) + 1
    steps_in_state = tuple(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    return StateGraph(
        nodes=tuple(states),
        edges=declared,
        visited_nodes=visited_nodes,
        visited_edges=visited_edges,
        unvisited_nodes=tuple(s for s in states if s not in set(visited_nodes)),
        unvisited_edges=tuple(e for e in declared if e not in set(visited_edges)),
        transitions=trans,
        path=tuple(path),
        steps_in_state=steps_in_state,
        n_transitions=len(trans),
    )


def state_path_of(graph: StateGraph) -> tuple[str, ...]:
    """状态路径的**去重保序**形式，用来一眼看出这次运行走了哪条路线。

    `path` 的最后一项已经是终态（最后一个转移发生在最后一个有转移的步上），
    所以这里不需要再补。
    """
    out: list[str] = []
    for _s, name in graph.path:
        if not out or out[-1] != name:
            out.append(name)
    return tuple(out)
