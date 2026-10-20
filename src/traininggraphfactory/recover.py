"""因果链复原：**从异常走到「发生了什么 / 何时 / 哪个资源 / 哪个配置 / 哪个 checkpoint」**。

三步走，第三步交给第三方：

| 阶段 | 做什么 | 依据 |
| --- | --- | --- |
| `0_locate` | 从异常出发，按声明的**反向传输边**走到可能的上游 | `causal_model` 的签名 |
| `1_recover` | 从声明的签名出发往回看事件，记录每条证据的**首次**满足步 | 检出结果 + 事件流 |
| `2_verify` | **交给第三方**：驱动真实模拟器重放，判定反事实 | `verify.py` |

**生成解释的人不给自己打分。** 所以本模块只产出 `Attribution`（答案），
一行真值都不读。真值在 `events._build_truths` 里，落进另一个产物文件。

两条判据（缺一不可）：

1. **签名命中率 ≥ `min_signature_hit`** —— 声明的签名里有几项真的在事件流里出现了。
2. **主信号证据必须在链上** —— 否则「发生了什么」这一问没有依据。
   这条判据专门挡掉一类假归因：第 60 步同时写 checkpoint 又跑评测，
   于是 `step_time_s` 与 `gpu0_util` 都被标出来，而 `vram_pressure` 的签名
   （`step_time_z` + `gpu_util_z_drop`）能凑到 2/3 = 0.67 通过第一条判据 ——
   但它缺 `vram_z`，所以被第二条挡下，如实报成「未归因」。

「首次」而不是「峰值」：一次持续 3 步的等比例跌幅里「哪一步是峰」是个平局，
让复原器去猜平局就等于让链的比对变成比运气。
"""

from __future__ import annotations

import bisect
from typing import Any, Iterable, Mapping, Sequence

from .config import causal_model_of, signatures_of
from .detect import DetectResult
from .align import assign_step
from .schema import (
    EVIDENCE_SIGNAL,
    FAILED_STATE_EDGE,
    ROOT_CAUSE_KINDS,
    Attribution,
    CausalHop,
    Event,
    StepSignal,
    chain_of,
    evidence_hop,
    resource_of_chain,
    what_of_chain,
)

__all__ = [
    "EvidenceIndex",
    "Candidate",
    "RecoverResult",
    "recover",
    "group_key",
    "primary_labels_of",
]


def primary_labels_of(primary_signal: str) -> tuple[str, ...]:
    """主信号对应哪些证据标签。`loss` 同时对应 `loss_z` 与 `loss_nan` ——
    这正是 `nan_grad` 能被认出来的原因：它的主信号也是 loss，但读数不是"高"而是"没了"。
    """
    return tuple(sorted(lab for lab, sig in EVIDENCE_SIGNAL.items() if sig == primary_signal))


class EvidenceIndex:
    """每个证据标签 → 它出现的步（升序）。

    **只吃事件流与检出结果。** 真值、`root_causes`、注入幅度一概不进这里 ——
    这是「复原器不能抄答案」的执行点。

    注意 `steps_by_event`：`io` 源不带 step 号，所以连「从哪个 checkpoint 恢复」
    这一条证据都必须先做时钟对齐才能落到某一步上。对齐错了，第五问就答错。
    """

    __slots__ = ("_steps", "ckpt_at_step")

    def __init__(self, run: Any, det: DetectResult, steps_by_event: Mapping[str, int]) -> None:
        steps: dict[str, set[int]] = {lab: set() for lab in EVIDENCE_SIGNAL}
        steps["config_change"] = set()
        steps["state_failed"] = set()
        steps["ckpt_restore"] = set()
        ckpt_at_step: dict[int, str] = {}

        # 信号类证据：来自检出（episode 的**起点**）
        for a in det.anomalies:
            for lab, sig in EVIDENCE_SIGNAL.items():
                if sig != a.signal:
                    continue
                if lab == "loss_nan":
                    if a.is_nan:
                        steps[lab].add(a.step_lo)
                    continue
                steps[lab].add(a.step_lo)

        # 事件类证据：来自 job / io 事件
        for e in run.events:
            st = steps_by_event.get(e.id)
            if st is None:
                continue
            if e.kind == "config_change" and e.source == "job":
                steps["config_change"].add(st)
            elif e.kind == "state_change" and str(e.payload.get("to")) == "FAILED":
                steps["state_failed"].add(st)
            elif e.kind == "read_ckpt":
                steps["ckpt_restore"].add(st)
                cid = str(e.payload.get("ckpt_id", ""))
                if cid:
                    ckpt_at_step[st] = cid

        self._steps = {k: tuple(sorted(v)) for k, v in steps.items()}
        self.ckpt_at_step = ckpt_at_step

    def steps(self, label: str) -> tuple[int, ...]:
        return self._steps.get(label, ())

    def first_in(self, label: str, lo: int, hi: int) -> int | None:
        """`[lo, hi]` 内该证据**首次**出现的步。"""
        xs = self._steps.get(label, ())
        i = bisect.bisect_left(xs, lo)
        if i < len(xs) and xs[i] <= hi:
            return xs[i]
        return None


class Candidate:
    """一个「这个异常是哪种根因」的候选。留着全部候选是为了让报告能解释为什么落选。"""

    __slots__ = ("kind", "hits", "missing", "steps", "ratio", "has_primary", "distance", "ok")

    def __init__(
        self,
        kind: str,
        hits: tuple[str, ...],
        missing: tuple[str, ...],
        steps: dict[str, int],
        ratio: float,
        has_primary: bool,
        distance: int,
        ok: bool,
    ) -> None:
        self.kind = kind
        self.hits = hits
        self.missing = missing
        self.steps = steps
        self.ratio = ratio
        self.has_primary = has_primary
        self.distance = distance
        self.ok = ok

    @property
    def sort_key(self) -> tuple[float, int, str]:
        # 命中率高的优先；打平时离异常近的优先；再打平按种类名（决定论）
        return (-self.ratio, self.distance, self.kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "hits": list(self.hits),
            "missing": list(self.missing),
            "steps": dict(self.steps),
            "ratio": self.ratio,
            "has_primary": self.has_primary,
            "distance": self.distance,
            "ok": self.ok,
        }


class RecoverResult:
    __slots__ = ("attributions", "unattributed", "candidates", "n_anomalies")

    def __init__(
        self,
        attributions: tuple[Attribution, ...],
        unattributed: tuple[str, ...],
        candidates: tuple[tuple[str, tuple[Candidate, ...]], ...],
        n_anomalies: int,
    ) -> None:
        self.attributions = attributions
        self.unattributed = unattributed
        self.candidates = candidates
        self.n_anomalies = n_anomalies

    def attribution(self, key: str) -> Attribution:
        for a in self.attributions:
            if group_key(a) == key:
                return a
        raise KeyError(f"没有这个归因：{key!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "attributions": [a.to_dict() for a in self.attributions],
            "unattributed": list(self.unattributed),
            "n_anomalies": self.n_anomalies,
            "candidates": [
                {"anomaly_id": aid, "candidates": [c.to_dict() for c in cands]} for aid, cands in self.candidates
            ],
        }


def group_key(a: Attribution) -> str:
    """一次根因的键：`<种类>@<何时>`。多条检出归到同一个键下就是一次异常。"""
    return f"{a.root_cause_kind}@{a.when_step}"


def resolve_steps(run: Any, al: Any | None) -> dict[str, int]:
    """`事件 id → 步`。

    带 step 号的事件直接用；其余（本项目里整个 `io` 源）按各自来源的**选中模型**
    反解回参考时钟，再用真实步边界查表。对齐结果缺失时，这些事件就没有步 ——
    不是猜一个 0，而是**不出现**。
    """
    if al is None:
        return {e.id: e.step for e in run.events if e.step is not None}
    bounds = al.step_boundaries
    starts = [b[1] for b in bounds]
    out: dict[str, int] = {}
    for e in run.events:
        if e.step is not None:
            out[e.id] = e.step
            continue
        name = al.chosen_model.get(e.source)
        if name is None:
            continue
        m = al.model(e.source, name)
        if m is None:
            continue
        st = assign_step((e.t_local - m.a) / m.b, starts, bounds)
        if st is not None:
            out[e.id] = st
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def recover(
    run: Any,
    step_table: Sequence[StepSignal],
    det: DetectResult,
    cfg: Mapping[str, Any],
    al: Any | None = None,
) -> RecoverResult:
    ccfg = cfg["causal"]
    lag = int(ccfg["lag_max_steps"])
    min_hit = float(ccfg["min_signature_hit"])
    tie_prox = bool(ccfg["tie_break_proximity"])
    prim_tol = int(ccfg.get("primary_tol_steps", 1))

    model = causal_model_of(cfg)
    sigs = signatures_of(cfg)
    ev = EvidenceIndex(run, det, resolve_steps(run, al))

    cfg_timeline = _config_timeline(run)
    ckpt_timeline = _ckpt_timeline(run)

    all_cands: list[tuple[str, tuple[Candidate, ...]]] = []
    best: dict[str, Candidate] = {}

    for a in det.anomalies:
        # 窗口**以 episode 的起点为锚**，不是峰值。取起点是因为「证据首次出现」
        # 这个约定本身就落在起点上；用峰值当锚，一次持续 8 步的异常会因为
        # 峰值落在第 5 步而把自己的起点排除在窗口外。
        lo, hi = a.step_lo - lag, a.step_lo + lag
        cands: list[Candidate] = []
        for kind in ROOT_CAUSE_KINDS:
            spec = model[kind]
            sig = sigs[kind]
            steps: dict[str, int] = {}
            for lab in sig:
                st = ev.first_in(lab, lo, hi)
                if st is not None:
                    steps[lab] = st
            hits = tuple(lab for lab in sig if lab in steps)
            missing = tuple(lab for lab in sig if lab not in steps)
            ratio = len(hits) / len(sig) if sig else 0.0
            prim = primary_labels_of(str(spec["primary_signal"]))
            prim_steps = [steps[lab] for lab in prim if lab in steps]
            # 主信号证据必须**贴近这次异常的起点**。只要求「出现过」是不够的：
            # 第 60 步（ckpt+eval 混合步）的假异常能靠 ±lag 的窗口够到第 62 步的
            # vram 证据，凑出一份看起来完美的签名，然后把真正的 vram_pressure
            # 的链污染成 `@55`。这条判据就是挡它的。
            distance = min((abs(s - a.step_lo) for s in prim_steps), default=10**6)
            has_primary = bool(prim_steps) and distance <= prim_tol
            ok = ratio >= min_hit and has_primary
            cands.append(Candidate(kind, hits, missing, steps, ratio, has_primary, distance, ok))
        cands.sort(key=lambda c: c.sort_key if tie_prox else (-c.ratio, 0, c.kind))
        all_cands.append((a.id, tuple(cands)))
        if cands and cands[0].ok:
            best[a.id] = cands[0]

    # 把指向同一个 `(种类, 何时)` 的检出并成一次归因
    grouped: dict[str, list[str]] = {}
    for aid, cand in best.items():
        key = f"{cand.kind}@{_when_of(cand, model)}"
        grouped.setdefault(key, []).append(aid)

    attributions: list[Attribution] = []
    for key in sorted(grouped, key=lambda k: (int(k.rsplit("@", 1)[1]), k)):
        aids = sorted(grouped[key])
        cand = best[aids[0]]
        kind = cand.kind
        spec = model[kind]
        sig = sigs[kind]
        # 同一组里各条检出的候选步骤可能略有出入（窗口不同）。取**并集里最早的**
        # 那个 —— 与真值链的「首次出现」约定一致。
        merged: dict[str, int] = {}
        for aid in aids:
            for lab, st in best[aid].steps.items():
                merged[lab] = min(st, merged.get(lab, st))
        hits = tuple(lab for lab in sig if lab in merged)
        missing = tuple(lab for lab in sig if lab not in merged)
        primary = str(spec["primary_signal"])
        when = _when_from_steps(merged, primary, spec)
        ckpt_key = ev.ckpt_at_step.get(merged.get("ckpt_restore", -1), "")
        chain = chain_of(
            [
                (
                    *evidence_hop(
                        lab,
                        merged[lab],
                        config_key=str(spec["config_key"]),
                        state_key=FAILED_STATE_EDGE,
                        ckpt_key=ckpt_key,
                    ),
                    merged[lab],
                )
                for lab in sig
                if lab in merged
            ]
        )
        attributions.append(
            Attribution(
                anomaly_id=aids[0],
                what=what_of_chain(primary, hits),
                when_step=when,
                resource=resource_of_chain(chain),
                config_key=str(spec["config_key"]),
                config_value=_config_value_at(cfg_timeline, str(spec["config_key"]), when),
                checkpoint=_checkpoint_before(ckpt_timeline, when),
                root_cause_kind=kind,
                chain=chain,
                signature_hits=hits,
                signature_missing=missing,
                covered_anomalies=tuple(aids),
                confidence=len(hits) / len(sig) if sig else 0.0,
                ambiguous=_is_ambiguous(all_cands, aids[0]),
            )
        )

    attributed = {aid for a in attributions for aid in a.covered_anomalies}
    unattributed = tuple(a.id for a in det.anomalies if a.id not in attributed)
    return RecoverResult(
        attributions=tuple(attributions),
        unattributed=unattributed,
        candidates=tuple(all_cands),
        n_anomalies=len(det.anomalies),
    )


def _when_of(cand: Candidate, model: Mapping[str, Mapping[str, Any]]) -> int:
    spec = model[cand.kind]
    return _when_from_steps(cand.steps, str(spec["primary_signal"]), spec)


def _when_from_steps(steps: Mapping[str, int], primary: str, spec: Mapping[str, Any]) -> int:
    prim = primary_labels_of(primary)
    cand = [steps[lab] for lab in prim if lab in steps]
    if not cand:
        return min(steps.values()) if steps else -1
    return min(cand)


def _is_ambiguous(all_cands: Sequence[tuple[str, tuple[Candidate, ...]]], aid: str) -> bool:
    for a_id, cands in all_cands:
        if a_id != aid:
            continue
        ok = [c for c in cands if c.ok]
        if len(ok) < 2:
            return False
        return abs(ok[0].ratio - ok[1].ratio) < 1e-9
    return False


def _config_timeline(run: Any) -> list[tuple[int, str, Any]]:
    """配置时间线：**从事件流读出来**，不是从 `cfg` 抄。

    `run_begin` 带上初始配置，之后每条 `config_change` 追加一条。于是「第 43 步
    当时用的是哪份配置」是个能被回答的问题 —— 而不是"去翻配置文件"。
    """
    out: list[tuple[int, str, Any]] = []
    for e in run.events:
        if e.source != "job":
            continue
        if e.kind == "run_begin":
            for k, v in sorted((e.payload.get("config") or {}).items()):
                out.append((int(e.step or 0), str(k), v))
        elif e.kind == "config_change":
            out.append((int(e.step or 0), str(e.payload["key"]), e.payload["value"]))
    return sorted(out, key=lambda t: (t[0], t[1]))


def _config_value_at(timeline: Sequence[tuple[int, str, Any]], key: str, step: int) -> Any:
    val: Any = None
    for st, k, v in timeline:
        if k == key and st <= step:
            val = v
    return val


def _ckpt_timeline(run: Any) -> list[tuple[int, str]]:
    """每个 checkpoint 是在哪一步写完的。**从事件读，不拿 `ckpt_every` 做除法** ——
    周期可能被改过，而「这一步真的写了 ckpt-0072」是事实。"""
    out: list[tuple[int, str]] = []
    for e in run.events:
        if e.source == "job" and e.kind == "ckpt_end" and e.step is not None:
            cid = str(e.payload.get("ckpt_id", ""))
            if cid:
                out.append((int(e.step), cid))
    return sorted(out)


def _checkpoint_before(timeline: Sequence[tuple[int, str]], step: int) -> str:
    """异常时刻「最近的那个 checkpoint」= 恢复点。第一个 checkpoint 之前如实返回空串。"""
    best = ""
    for st, cid in timeline:
        if st <= step:
            best = cid
    return best
