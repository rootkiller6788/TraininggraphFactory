"""抽取血缘 DAG。

这个项目的血缘**不是数据的血缘，是信号的血缘**：一条 `evidence` 边从哪一层来、
一条 `attribution` 边从哪几个异常来。于是有两条别处没有的图不变量：

1. **`attrib:<kind>@<step>` 的父节点恰好是它 `covered_anomalies` 里那几个检出。**
   一个不多（没把无关的异常混进来）、一个不少（每个被覆盖的异常都被接住了）。
   这条不变量把「一次异常被几个检出碎片描述」这件事**画进了图里** ——
   报告里只看归因表，看不出「这次归因到底吃掉了几条检出」。
2. **`hop:<layer>:<key>` 的父节点恰好是链上前一跳。** 于是从任意一跳沿着
   `parents` 往上走，走到根就是这条链的**第一因**；这与 `chain` 的顺序
   互为校验 —— 一处写错，另一处对不上。

`lineage_chain.dot` 只画**因果链**（8 层信号 → 6 个归因 → 五问答案），
不画事件流 —— 120 步 × 4 路来源有 2000+ 个事件，全画出来没人看得清。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema import LAYERS, utcnow

__all__ = [
    "Node",
    "Edge",
    "Lineage",
    "RUN_NODE_ID",
    "EVENTS_NODE_ID",
    "ALIGN_NODE_ID",
    "DETECT_NODE_ID",
    "RECOVER_NODE_ID",
    "VERIFY_NODE_ID",
    "ANSWER_NODE_ID",
    "run_node",
    "source_node",
    "event_node",
    "signal_node",
    "graph_node",
    "anomaly_node",
    "hop_node",
    "attribution_node",
    "answer_node",
    "build_lineage",
    "dumps",
    "write",
]


RUN_NODE_ID = "run"
EVENTS_NODE_ID = "events"
ALIGN_NODE_ID = "align"
DETECT_NODE_ID = "detect"
RECOVER_NODE_ID = "recover"
VERIFY_NODE_ID = "verify"
ANSWER_NODE_ID = "answer"


def run_node() -> str:
    return RUN_NODE_ID


def source_node(source: str) -> str:
    return f"source:{source}"


def event_node(event_id: str) -> str:
    return f"event:{event_id}"


def signal_node(step: int) -> str:
    return f"signal:step-{step:04d}"


def graph_node(which: str) -> str:
    """三张图各是一个节点。`which ∈ {job, state, resource}` —— **三张图必须分开**，
    合成一个节点就等于把「数据层就分开」这条设计又收了回去。"""
    if which not in ("job", "state", "resource"):
        raise ValueError(f"未知的图：{which!r}（可选：job / state / resource）")
    return f"graph:{which}"


def anomaly_node(anomaly_id: str) -> str:
    return f"anomaly:{anomaly_id}"


def hop_node(layer: str, key: str) -> str:
    if layer not in LAYERS:
        raise ValueError(f"未声明的层：{layer!r}")
    return f"hop:{layer}:{key}"


def attribution_node(kind: str, step: int) -> str:
    return f"attrib:{kind}@{step}"


def answer_node(kind: str) -> str:
    return f"answer:{kind}"


@dataclass
class Node:
    id: str
    kind: str
    label: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "label": self.label, "meta": self.meta}


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "label": self.label}


class Lineage:
    """信号血缘图。`add_node` 拒重复、`add_edge` 校验两端存在。"""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: set[tuple[str, str, str]] = set()
        # 邻接索引。`parents` / `children` 原来每次都扫全边集，于是 `roots` /
        # `leaves` / `path_to_root` 都是 O(节点 × 边)。索引只在 `add_edge`（唯一的
        # 加边入口）里同步，两边不会漂。
        self._parents: dict[str, set[str]] = {}
        self._children: dict[str, set[str]] = {}

    def add_node(self, node_id: str, kind: str, label: str = "", **meta: Any) -> Node:
        if node_id in self._nodes:
            raise ValueError(f"节点已存在：{node_id}")
        n = Node(id=node_id, kind=kind, label=label or node_id, meta=dict(meta))
        self._nodes[node_id] = n
        return n

    def add_edge(self, src: str, dst: str, label: str = "") -> None:
        for end in (src, dst):
            if end not in self._nodes:
                raise ValueError(f"边 {src}->{dst} 的端点 {end!r} 不在图里")
        self._edges.add((src, dst, label))
        self._parents.setdefault(dst, set()).add(src)
        self._children.setdefault(src, set()).add(dst)

    # -- 查询 --------------------------------------------------------------

    def has(self, node_id: str) -> bool:
        return node_id in self._nodes

    def node(self, node_id: str) -> Node:
        if node_id not in self._nodes:
            raise KeyError(f"没有节点 {node_id!r}")
        return self._nodes[node_id]

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes[k] for k in sorted(self._nodes))

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(Edge(*t) for t in sorted(self._edges))

    def parents(self, node_id: str) -> tuple[str, ...]:
        return tuple(sorted(self._parents.get(node_id, ())))

    def children(self, node_id: str) -> tuple[str, ...]:
        return tuple(sorted(self._children.get(node_id, ())))

    def roots(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.nodes if not self.parents(n.id))

    def leaves(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.nodes if not self.children(n.id))

    def by_kind(self, kind: str) -> tuple[Node, ...]:
        return tuple(n for n in self.nodes if n.kind == kind)

    def path_to_root(self, node_id: str) -> tuple[str, ...]:
        """任取一条到根的路径（含两端）。"""
        path = [node_id]
        seen = {node_id}
        cur = node_id
        while True:
            ps = self.parents(cur)
            if not ps:
                return tuple(path)
            nxt = ps[0]
            if nxt in seen:
                raise ValueError(f"图里有环：{path} → {nxt}")
            path.append(nxt)
            seen.add(nxt)
            cur = nxt

    def depth(self, node_id: str) -> int:
        return len(self.path_to_root(node_id)) - 1

    # -- 导出 --------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Lineage":
        lin = cls()
        for n in d["nodes"]:
            lin.add_node(n["id"], n["kind"], n["label"], **n.get("meta", {}))
        for e in d["edges"]:
            lin.add_edge(e["src"], e["dst"], e.get("label", ""))
        return lin

    def to_dot(self, *, include: Sequence[str] | None = None) -> str:
        keep = set(include) if include is not None else set(self._nodes)
        lines = [
            "digraph traininggraphfactory {",
            '  rankdir="LR";',
            '  node [shape=box, fontname="Helvetica"];',
        ]
        for n in self.nodes:
            if n.id not in keep:
                continue
            lines.append(f'  "{n.id}" [label="{n.label}"];')
        for e in self.edges:
            if e.src not in keep or e.dst not in keep:
                continue
            lab = f' [label="{e.label}"]' if e.label else ""
            lines.append(f'  "{e.src}" -> "{e.dst}"{lab};')
        lines.append("}")
        return "\n".join(lines) + "\n"

    def to_ascii(self, *, limit: int = 40) -> str:
        lines: list[str] = []
        for n in self.nodes[:limit]:
            ps = self.parents(n.id)
            dep = "  " * min(self.depth(n.id), 6)
            lines.append(f"{dep}{n.label}  [{n.kind}]")
            if ps:
                lines.append(f"{dep}  ← {', '.join(ps)}")
        if len(self.nodes) > limit:
            lines.append(f"… 还有 {len(self.nodes) - limit} 个节点（用 --dot 看全图）")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 组装
# --------------------------------------------------------------------------- #


def build_lineage(
    *,
    run: Any,
    al: Any,
    det: Any,
    rec: Any,
    sources: Sequence[str],
    n_steps: int,
    graphs: Mapping[str, int],
) -> Lineage:
    """搭出信号血缘 DAG。

    形状是**扇入**：2000+ 个事件 → 120 张逐步信号表 → 20 条检出 → 6 个归因 → 6 个答案。
    事件层只挂**每个来源一个**代表节点，不逐个事件建节点 —— 这不是偷懒：
    逐事件建节点会让图有 2000 个叶子，而 `path_to_root` 就成了随机漫步。
    真正要能追溯的是「哪一层的信号进了哪条链」。
    """
    lin = Lineage()
    lin.add_node(RUN_NODE_ID, "run", f"Training Run（{n_steps} 步）", n_steps=n_steps)

    # 事件流：每个来源一个节点，边标出该来源产出了多少个事件。
    lin.add_node(EVENTS_NODE_ID, "stream", f"Event Stream（{len(run.events)} 个事件）")
    for s in sources:
        evs = run.events_of(s)
        lin.add_node(source_node(s), "source", f"source:{s}（{len(evs)} 个事件）", n=len(evs))
        lin.add_edge(EVENTS_NODE_ID, source_node(s), "分路")

    lin.add_node(ALIGN_NODE_ID, "stage", "align（四路 → 逐步信号表）")
    lin.add_edge(EVENTS_NODE_ID, ALIGN_NODE_ID, "对齐")

    # 逐步信号表：120 个节点，各自挂在对齐节点下。这是**唯一**一层逐点的扇出，
    # 因为三张图与检出都要靠 step 号把它们接起来。
    for st in al.step_table:
        nid = signal_node(st.step)
        lin.add_node(nid, "signal", f"step-{st.step:04d}", n_sources=st.n_sources)
        lin.add_edge(ALIGN_NODE_ID, nid, "逐步")

    # 三张图。**分开建、分开挂** —— 合起来就等于把设计收回去。
    for which in ("job", "state", "resource"):
        nid = graph_node(which)
        lin.add_node(nid, "graph", f"{which} graph（{graphs[which]}）")
        if which == "job":
            lin.add_edge(RUN_NODE_ID, nid, "job 事件")
        else:
            lin.add_edge(ALIGN_NODE_ID, nid, "逐步信号表")

    lin.add_node(DETECT_NODE_ID, "stage", f"detect（{len(det.anomalies)} 条检出）")
    lin.add_edge(ALIGN_NODE_ID, DETECT_NODE_ID, "分层基线")

    for a in det.anomalies:
        nid = anomaly_node(a.id)
        lin.add_node(nid, "anomaly", f"{a.signal}@{a.step_lo}-{a.step_hi}", signal=a.signal)
        lin.add_edge(DETECT_NODE_ID, nid, "检出")
        # 逐步信号表 → 异常：只连它自己的那几步。**这是扇出里唯一的一处精确连边**，
        # 少了它，血缘就退化成「一张表里恰好有这些数」。
        for st in range(a.step_lo, a.step_hi + 1):
            if lin.has(signal_node(st)):
                lin.add_edge(signal_node(st), nid, a.signal)

    lin.add_node(RECOVER_NODE_ID, "stage", f"recover（{len(rec.attributions)} 起归因）")
    lin.add_edge(DETECT_NODE_ID, RECOVER_NODE_ID, "签名匹配")

    lin.add_node(ANSWER_NODE_ID, "answer", "五问答案（what / when / resource / config / ckpt）")
    lin.add_node(VERIFY_NODE_ID, "stage", "verify（第三方重放）")
    lin.add_edge(ANSWER_NODE_ID, VERIFY_NODE_ID, "受检")

    for at in rec.attributions:
        aid = attribution_node(at.root_cause_kind, at.when_step)
        lin.add_node(aid, "attribution", f"{at.root_cause_kind}@{at.when_step}", confidence=at.confidence)
        # 不变量 1：父节点恰好是 covered_anomalies。**不是「相关的检出」**。
        for an_id in at.covered_anomalies:
            if lin.has(anomaly_node(an_id)):
                lin.add_edge(anomaly_node(an_id), aid, "覆盖")
        # 因果链：逐跳接起来，边标出层名。
        prev: str | None = None
        for h in at.chain:
            nid = hop_node(h.layer, h.key)
            if not lin.has(nid):
                lin.add_node(nid, "hop", f"{h.layer}:{h.key}", layer=h.layer, step=h.step)
            lin.add_edge(nid, aid, "证据")
            # 不变量 2：链上前一跳 → 这一跳。`hop` 节点可能被两条链共用
            # （比如两条链都经过 `metric:loss`），共用就共用 —— 边是幂等的。
            if prev is not None:
                lin.add_edge(prev, nid, "→")
            prev = nid

        ans = answer_node(at.root_cause_kind)
        if not lin.has(ans):
            lin.add_node(ans, "answer", f"answer:{at.root_cause_kind}")
        lin.add_edge(aid, ans, "回答")
        lin.add_edge(ans, ANSWER_NODE_ID, "五问")

    for at in rec.attributions:
        lin.add_edge(attribution_node(at.root_cause_kind, at.when_step), VERIFY_NODE_ID, "受检")
    return lin


def dumps(lin: Lineage) -> str:
    return json.dumps(lin.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write(lin: Lineage, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(lin), encoding="utf-8")
    return path


def chain_hop_ids(chain: Iterable[Any]) -> tuple[str, ...]:
    """一条链的跳节点 id。给 `--dot` 的 include 用。"""
    return tuple(hop_node(h.layer, h.key) for h in chain)
