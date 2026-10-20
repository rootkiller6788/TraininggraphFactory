"""自包含 HTML：**零外部资源**。

没有 CDN、没有字体、没有图片、没有 `fetch`。一个报告要在别人的机器上打开就
必须自己带齐全部东西 —— 而「打开时去要一个外网字体」在断网的机器上等于白屏。

三张图**分成三块画**，不是画在一张图上。这不是排版选择：把 Job DAG、状态图、
资源图叠进一个 SVG，看的人第一件事就是去找「哪个框属于哪张图」——
那正是 README 第三节说的「退化成一张杂乱的流程图」。
"""

from __future__ import annotations

import html
import math
from typing import Any, Mapping, Sequence

from .recover import group_key
from .schema import LAYERS, SIGNALS

__all__ = ["render_html"]

#: 八层信号的颜色。**顺序与 `LAYERS` 一致** —— 图上从上到下就是因到果。
_LAYER_COLOR = {
    "config": "#8250df",
    "data": "#0969da",
    "io": "#0a7ea4",
    "job": "#1a7f37",
    "resource": "#9a6700",
    "metric": "#cf222e",
    "state": "#6e7781",
    "checkpoint": "#bf3989",
}

_CSS = """
:root{
  --bg:#ffffff; --panel:#f6f8fa; --line:#d0d7de; --ink:#1f2328; --muted:#656d76;
  --accent:#0969da; --ok:#1a7f37; --bad:#cf222e; --warn:#9a6700;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:26px;margin:0 0 6px}
h2{font-size:19px;margin:38px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:22px 0 8px;color:var(--muted);font-weight:600}
.sub{color:var(--muted);font-size:13px;margin-bottom:18px}
.sub code{background:var(--panel);padding:1px 5px;border-radius:4px}
.note{background:var(--panel);border-left:3px solid var(--accent);padding:10px 14px;
  margin:14px 0;font-size:13px;color:var(--muted)}
.note b{color:var(--ink)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;margin:12px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{background:var(--panel);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px}
.mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px}
.ok{color:var(--ok);font-weight:600}
.bad{color:var(--bad);font-weight:600}
.warn{color:var(--warn);font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--muted);margin:6px 0 0}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:10px;margin:12px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.card .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:20px;font-weight:600;font-variant-numeric:tabular-nums}
.card .d{font-size:11px;color:var(--muted)}
svg{display:block}
""".strip()


def _e(x: Any) -> str:
    return html.escape(str(x), quote=True)


def _fmt(x: Any, *, sig: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "是" if x else "否"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf"
        return f"{x:.{sig}g}"
    return str(x)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], *, numeric: Sequence[int] = ()) -> str:
    th = "".join(
        f'<th class="{"num" if i in numeric else ""}">{_e(h)}</th>' for i, h in enumerate(header)
    )
    trs = []
    for r in rows:
        tds = "".join(
            f'<td class="{"num" if i in numeric else ""}">{c}</td>' for i, c in enumerate(r)
        )
        trs.append(f"<tr>{tds}</tr>")
    return f'<table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


# --------------------------------------------------------------------------- #
# 图一：Job DAG
# --------------------------------------------------------------------------- #


def _job_svg(result: Any) -> str:
    job = result.job_dag
    tpl_nodes = [t for t, _l in job.template_nodes]
    tpl_edges = job.template_edges
    # 层号取**最长路径**深度，不是「拓扑序里第一次出现的位置」：
    # 后者会让 `export`（同时依赖 `train` 与 `checkpoint`）落在 checkpoint 那一列上，
    # 于是两条边画成一条水平线，看的人会以为它们同级。
    # **自环必须排除**：`train → train`（相邻实例串起来）在松弛里会把
    # `level[train]` 每轮加 1，9 轮之后模板被推到第 9 列、图宽 2980px。
    level: dict[str, int] = {t: 0 for t in tpl_nodes}
    for _ in range(len(tpl_nodes) + 1):
        for a, b in tpl_edges:
            if a == b:
                continue
            if level[b] < level[a] + 1:
                level[b] = level[a] + 1
    cols: dict[int, list[str]] = {}
    for t in tpl_nodes:
        cols.setdefault(level[t], []).append(t)
    n_cols = max(cols) + 1 if cols else 1
    n_rows = max(len(v) for v in cols.values()) if cols else 1

    BW, BH, GAP_X, GAP_Y = 132, 34, 78, 18
    W = 40 + n_cols * (BW + GAP_X)
    H = 46 + n_rows * (BH + GAP_Y)

    pos: dict[str, tuple[float, float]] = {}
    for d, members in sorted(cols.items()):
        for i, t in enumerate(sorted(members)):
            pos[t] = (40 + d * (BW + GAP_X), 40 + i * (BH + GAP_Y))

    parts: list[str] = [f'<svg viewBox="0 0 {W} {H}" width="100%" height="{H}" role="img" preserveAspectRatio="xMinYMin meet">']
    parts.append(
        f'<text x="8" y="20" font-size="12" fill="#656d76">'
        f'模板：{len(job.template_nodes)} 节点 / {len(job.template_edges)} 边 ｜ '
        f'实例：{len(job.nodes)} 节点 / {len(job.edges)} 边</text>'
    )
    parts.append(
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
        'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#8c959f"/></marker></defs>'
    )
    for a, b in tpl_edges:
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        if a == b:
            # 自环（`train → train`：同一个模板的相邻实例串起来）。
            # 画在节点上方 —— 走「同列向下」那条分支会得到一条退化成点的曲线。
            parts.append(
                f'<path d="M {x1 + BW - 34} {y1} C {x1 + BW + 14} {y1 - 34}, {x1 + BW - 2} {y1 - 34}, '
                f'{x1 + BW - 30} {y1 - 4}" fill="none" stroke="#8c959f" stroke-width="1.4" '
                f'marker-end="url(#arrow)"/>'
            )
            continue
        if abs(x2 - x1) < 1.0:
            # 同列不同行：走左侧绕回去。
            parts.append(
                f'<path d="M {x1} {y1 + BH / 2} C {x1 - 40} {y1 + BH / 2}, '
                f'{x2 - 40} {y2 + BH / 2}, {x2} {y2 + BH / 2}" fill="none" stroke="#8c959f" '
                f'stroke-width="1.4" marker-end="url(#arrow)"/>'
            )
            continue
        parts.append(
            f'<path d="M {x1 + BW} {y1 + BH / 2} C {x1 + BW + 40} {y1 + BH / 2}, '
            f'{x2 - 40} {y2 + BH / 2}, {x2} {y2 + BH / 2}" fill="none" stroke="#8c959f" '
            f'stroke-width="1.4" marker-end="url(#arrow)"/>'
        )
    counts: dict[str, int] = {}
    for n in job.nodes:
        counts[n.template] = counts.get(n.template, 0) + 1
    for tpl in tpl_nodes:
        x, y = pos[tpl]
        c = counts.get(tpl, 0)
        fill = "#ddf4e4" if c else "#f6f8fa"
        parts.append(
            f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="6" fill="{fill}" '
            f'stroke="#1a7f37" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{x + 8}" y="{y + 15}" font-size="11" fill="#1f2328">{_e(tpl)}</text>'
            f'<text x="{x + 8}" y="{y + 28}" font-size="10" fill="#656d76">×{c} 实例</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# 图二：State Graph
# --------------------------------------------------------------------------- #


def _state_svg(result: Any) -> str:
    stg = result.state_graph
    nodes = list(stg.nodes)
    idx = {s: i for i, s in enumerate(nodes)}
    W, H = 980, 210
    parts: list[str] = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">']
    parts.append(
        f'<text x="8" y="18" font-size="12" fill="#656d76">'
        f'声明 {len(stg.nodes)} 状态 / {len(stg.edges)} 边 ｜ 走过 {len(stg.visited_nodes)} 状态 / '
        f'{len(stg.visited_edges)} 边 ｜ 没走到：{_e(", ".join(stg.unvisited_nodes) or "（无）")}</text>'
    )
    visited = set(stg.visited_nodes)
    vedges = set(stg.visited_edges)
    r = 34
    span = (W - 120) / max(1, len(nodes) - 1)
    cx = {s: 70 + i * span for i, s in enumerate(nodes)}
    cy = {s: 110 for s in nodes}

    for a, b in stg.edges:
        x1, x2 = cx[a], cx[b]
        y1, y2 = cy[a], cy[b]
        on = (a, b) in vedges
        col = "#1a7f37" if on else "#c8ced6"
        if abs(x2 - x1) < 1.0:
            # 自环：画在节点上方
            parts.append(
                f'<path d="M {x1 - 12} {y1 - r + 4} C {x1 - 40} {y1 - r - 46}, {x1 + 40} {y1 - r - 46}, '
                f'{x1 + 12} {y1 - r + 4}" fill="none" stroke="{col}" stroke-width="1.4"/>'
            )
            continue
        d = 1 if x2 > x1 else -1
        # `stroke-dasharray` 的值**必须带引号** —— 不引的话 HTML 能显示，
        # 但它不是合法 XML，任何拿 XML 解析器读这份 SVG 的人都会当场失败。
        dash = '' if on else ' stroke-dasharray="4,3"'
        parts.append(
            f'<path d="M {x1 + d * (r + 3)} {y1} C {(x1 + x2) / 2} {y1 - 58}, {(x1 + x2) / 2} {y2 - 58}, '
            f'{x2 - d * (r + 6)} {y2}" fill="none" stroke="{col}" stroke-width="1.4"{dash}/>'
        )
    for s in nodes:
        x, y = cx[s], cy[s]
        on = s in visited
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{"#ddf4e4" if on else "#ffffff"}" '
            f'stroke="{"#1a7f37" if on else "#c8ced6"}" stroke-width="{2 if on else 1.2}"/>'
        )
        parts.append(
            f'<text x="{x}" y="{y + 4}" font-size="11" text-anchor="middle" '
            f'fill="{"#1f2328" if on else "#8c959f"}">{_e(s)}</text>'
        )
    parts.append(
        f'<text x="8" y="{H - 12}" font-size="12" fill="#656d76">'
        f'路径：{_e(" → ".join(s for _st, s in stg.path))}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# 图三：Resource Graph
# --------------------------------------------------------------------------- #


def _resource_svg(result: Any) -> str:
    rg = result.resource_graph
    res = list(rg.resources)
    W = 980
    row_h = 30
    H = row_h * len(res) + 60
    parts: list[str] = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">']
    parts.append(
        f'<text x="8" y="18" font-size="12" fill="#656d76">'
        f'中枢 {_e(rg.hub)} ｜ 归一化用**该资源自己的稳健 z** ｜ '
        f'瓶颈只在压力超过 k 时报，本次 {len(rg.bottleneck_by_step)} 步</text>'
    )
    ranked = rg.ranked()
    top = max((p.normalized for p in ranked), default=1.0) or 1.0
    x0, x1 = 240, W - 120
    for i, p in enumerate(ranked):
        y = 34 + i * row_h
        w = max(1.0, (p.normalized / top) * (x1 - x0)) if p.normalized > 0 else 0.0
        bad = p.normalized > float(result.cfg["detect"]["k"])
        col = "#cf222e" if bad else "#0969da"
        parts.append(
            f'<text x="8" y="{y + 13}" font-size="12" fill="#1f2328">{_e(p.resource)}</text>'
            f'<text x="120" y="{y + 13}" font-size="11" fill="#656d76" class="mono">{_e(rg.signal_of.get(p.resource, ""))}</text>'
        )
        parts.append(f'<rect x="{x0}" y="{y + 3}" width="{x1 - x0}" height="16" rx="3" fill="#eef1f4"/>')
        if w > 0:
            parts.append(f'<rect x="{x0}" y="{y + 3}" width="{w:.1f}" height="16" rx="3" fill="{col}"/>')
        parts.append(
            f'<text x="{x1 + 8}" y="{y + 16}" font-size="11" fill="#656d76" class="mono">'
            f'z={_fmt(p.normalized, sig=3)} @{p.peak_step}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# 因果链泳道图
# --------------------------------------------------------------------------- #

_HOP_SHORT = {
    "config_change": "config",
    "state_failed": "state",
    "ckpt_restore": "ckpt",
}


def _chain_svg(result: Any, attrs: Sequence[Any]) -> str:
    if not attrs:
        return "<svg viewBox='0 0 980 40' width='980' height='40'></svg>"
    lane_h = 30
    gap = 26
    W = 980
    H = len(attrs) * (lane_h * 1 + gap) + 40
    parts: list[str] = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img">']
    parts.append(
        f'<text x="8" y="16" font-size="12" fill="#656d76">'
        f'横向是链上的跳（从因到果），颜色是层；`{_e("/".join(LAYERS))}`</text>'
    )
    all_steps: list[int] = []
    for a in attrs:
        all_steps.extend(h.step for h in a.chain)
    lo, hi = (min(all_steps), max(all_steps)) if all_steps else (0, 1)
    span = max(1, hi - lo)
    x0, x1 = 250, W - 60

    for i, a in enumerate(attrs):
        y = 34 + i * (lane_h + gap)
        parts.append(
            f'<text x="8" y="{y + 18}" font-size="12" fill="#1f2328">{_e(group_key(a))}</text>'
            f'<text x="8" y="{y + 32}" font-size="10" fill="#656d76">'
            f'{_e(a.resource)} ｜ conf={_fmt(a.confidence, sig=2)}</text>'
        )
        parts.append(
            f'<line x1="{x0}" y1="{y + 18}" x2="{x1}" y2="{y + 18}" stroke="#e3e6ea" stroke-width="1"/>'
        )
        pts: list[tuple[float, float]] = []
        for h in a.chain:
            px = x0 + ((h.step - lo) / span) * (x1 - x0)
            py = y + 18
            pts.append((px, py))
        if len(pts) > 1:
            parts.append(
                '<path d="' + " ".join(f'{"M" if k == 0 else "L"} {px:.1f} {py:.1f}' for k, (px, py) in enumerate(pts))
                + '" fill="none" stroke="#8c959f" stroke-width="1.4"/>'
            )
        for h, (px, py) in zip(a.chain, pts):
            col = _LAYER_COLOR.get(h.layer, "#6e7781")
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="{col}"/>')
            short = h.key.split(":")[-1].split("@")[0]
            parts.append(
                f'<text x="{px:.1f}" y="{py - 11}" font-size="10" fill="{col}" text-anchor="middle">{_e(short)}</text>'
            )
            parts.append(
                f'<text x="{px:.1f}" y="{py + 21}" font-size="9" fill="#8c959f" text-anchor="middle">{h.step}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #


def render_html(result: Any, *, rows: Sequence[Mapping[str, Any]] | None = None) -> str:
    from .report import acceptance_checks, expectation_rows

    if rows is None:
        rows = expectation_rows(result)

    al = result.alignment
    det = result.detect_result
    rec = result.recover_result
    ver = result.verify_result
    acc = acceptance_checks(result)
    n_acc = sum(1 for a in acc if a["passed"])
    refuted = [r for r in rows if not r["held"]]

    P: list[str] = []
    P.append("<!DOCTYPE html>")
    P.append('<html lang="zh-CN"><head><meta charset="utf-8">')
    P.append('<meta name="viewport" content="width=device-width,initial-scale=1">')
    P.append("<title>TrainingGraphFactory 归因报告</title>")
    P.append(f"<style>{_CSS}</style></head><body><div class='wrap'>")

    P.append("<h1>TrainingGraphFactory 归因报告</h1>")
    # 与 `report.md` 同一个理由：正文不印 `created_at`，否则两次相同输入的报告不逐位相同。
    P.append(
        f"<div class='sub'>版本 <code>{_e(result.version.id)}</code> ｜ "
        f"口径 <code>{_e(result.version.protocol_digest[:12])}</code> ｜ "
        f"{result.version.n_steps} 步 / {result.version.n_events} 个事件 ｜ 生成时间见 <code>version.json</code></div>"
    )
    P.append(
        "<div class='note'><b>这不是 Training UI。</b> 它是 Training Runtime 的数字孪生 —— "
        "不画曲线，而是<b>在五张不同的信号之间建立可追溯的因果链</b>。"
        "任何一次异常都能回答：发生了什么 / 何时 / 哪个资源 / 哪个配置 / 哪个 checkpoint。</div>"
    )

    # 卡片
    P.append("<div class='cards'>")
    cards = [
        ("归因", f"{len(rec.attributions)}/{len(result.run.truths)}", f"未归因检出 {len(rec.unattributed)} 条"),
        ("五问", f"{_pct(_rate(result, 'five'))}", "五问全对的比例"),
        ("因果链", f"{_pct(_rate(result, 'chain'))}", "链逐键一致的比例"),
        ("验收", f"{n_acc}/{len(acc)}", "门槛全部满足" if n_acc == len(acc) else "有未通过项"),
        ("P/F/h", " / ".join(_pct(ver.probe_pass_rate(p)) for p in ("P", "F", "h")), "三向探针通过率"),
        ("对齐 lift", _pct(ver.alignment_lift()), "漂移感知多救回的归因"),
        ("被推翻的期望", str(len(refuted)), "允许期望被推翻"),
    ]
    for k, v, d in cards:
        cls = ""
        if k == "验收" and n_acc != len(acc):
            cls = " bad"
        elif k == "被推翻的期望" and refuted:
            cls = " warn"
        P.append(f"<div class='card'><div class='k'>{_e(k)}</div><div class='v{cls}'>{_e(v)}</div><div class='d'>{_e(d)}</div></div>")
    P.append("</div>")

    # 一、三张图
    P.append("<h2>一、三张图：分开构建、分开查询</h2>")
    P.append(
        "<div class='note'>三者混在一起就会退化成一张杂乱的流程图，"
        "所以它们在<b>数据层就分开</b> —— 也落成三份产物、画成三块。</div>"
    )
    P.append("<h3>1 · Job DAG</h3><div class='panel'>")
    P.append(_job_svg(result))
    P.append("</div>")
    job = result.job_dag
    P.append(
        f"<div class='sub'>任取一步查依赖：<code>checkpoint@72</code> ← "
        f"<code>{_e(', '.join(job.deps('checkpoint@72')) or '（无依赖）')}</code>，"
        f"祖先 {len(job.ancestors('checkpoint@72'))} 个。"
        "拓扑序在入度为 0 时按节点 id 排序 —— 否则会依赖 <code>set</code> 的迭代顺序。</div>"
    )

    P.append("<h3>2 · Training State Graph</h3><div class='panel'>")
    P.append(_state_svg(result))
    P.append("</div>")
    P.append(
        "<div class='note'>没走到的状态与转移<b>照样画出来</b>（灰色）："
        "「声明了但这次没发生」与「声明里根本没有」必须分开报。</div>"
    )

    P.append("<h3>3 · Resource Graph</h3><div class='panel'>")
    P.append(_resource_svg(result))
    P.append("</div>")

    # 二、对齐
    P.append("<h2>二、对齐：四路异质事件 → 一张逐步信号表</h2>")
    P.append(
        f"<div class='note'>只有 <code>job</code> 带步号，其余三路必须靠<b>同步信标</b>接回来。"
        f"共 {al.n_events_total} 个待归属事件，{al.n_events_assigned} 个接上了步。"
        "判据是<b>步归属错误率</b>而不是拟合优度。</div>"
    )
    rows_a: list[list[str]] = []
    for r in al.reports:
        by_name = {m.model: m for m in al.models_of(r.source)}
        for name in ("identity", "constant", "linear"):
            m = by_name.get(name)
            if m is None:
                continue
            mark = ' <span class="ok">←</span>' if r.model == name else ""
            rows_a.append(
                [
                    f"<code>{_e(r.source)}</code>" if name == "identity" else "",
                    f"<code>{_e(name)}</code>{mark}",
                    _fmt(m.a, sig=5),
                    _fmt(m.b, sig=6),
                    _fmt(m.residual_rms_s, sig=3),
                    _pct(m.assign_error_rate),
                ]
            )
    P.append(_table(("来源", "假设", "估计偏移 (s)", "估计斜率", "残差 RMS (s)", "步归属错误率"), rows_a, numeric=(2, 3, 4, 5)))
    P.append(
        "<div class='note'>① 残差必须在容差内（<code>identity</code> 就是这样出局的 —— 它的残差等于偏移本身）；"
        "② 在剩下的里挑错误率最低的，<b>打平则取更简单的</b> —— <code>linear</code> 必须挣得它的第二个参数。</div>"
    )

    # 三、检出
    P.append("<h2>三、检出：去趋势 + 分层 + 乘性通道</h2>")
    by_sig = det.by_signal()
    P.append(
        _table(
            ("信号", "条数", "步区间", "峰值 z"),
            [
                (
                    f"<code>{_e(s)}</code>",
                    str(len(v)),
                    "、".join(f"{a.step_lo}-{a.step_hi}" if a.step_hi > a.step_lo else str(a.step_lo) for a in v),
                    _fmt(max((a.peak_z for a in v if not a.is_nan), default=float("nan")), sig=3),
                )
                for s, v in sorted(by_sig.items())
            ],
            numeric=(1, 3),
        )
    )
    P.append(
        "<div class='note'><b>为什么必须去趋势：</b><code>loss</code> 从 4.2 衰减到 0.12（34×），"
        "全局中位数基线检不出真正的 spike，所以用 Theil–Sen 去趋势、跨度 ≥ 4 的信号走对数通道。<br>"
        "<b>为什么必须分层：</b><code>ckpt_every=12</code> 与 <code>eval_every=10</code> 每 60 步撞一次，"
        "第 60 步既写 ckpt 又跑 eval，耗时比普通 ckpt 步长 30% —— 在 ckpt 层里是 z≈15 的假阳性。<br>"
        "<b>NaN 必须显式判异常：</b><code>abs(nan - median) &gt; k*mad</code> 永远是 False。</div>"
    )
    if det.skipped:
        P.append(
            f"<div class='note'><b>未判定</b>：{len(det.skipped)} 个分层因为样本不足而没有判定"
            f"（<code>{_e(', '.join(f'{k}:{v}' for k, v in sorted(det.skipped.items())))}</code>）。"
            "「未判定」与「正常」必须分开报。</div>"
        )

    # 四、归因
    P.append("<h2>四、归因：五问</h2>")
    P.append(
        "<div class='note'>归因与真值<b>分两个文件</b>：归因器一行真值都不读，"
        "<code>truths.json</code> 是第三方（<code>verify</code>）用的。"
        "证据步的约定是<b>「首次出现」而不是峰值</b> —— 一次持续 3 步的等比例跌幅里"
        "「哪一步是峰」是平局，让两边各自猜平局等于让链的比对变成比运气。</div>"
    )
    P.append(
        _table(
            ("根因", "what", "when", "resource", "config", "checkpoint", "置信度"),
            [
                (
                    f"<code>{_e(group_key(a))}</code>",
                    _e(a.what),
                    str(a.when_step),
                    _e(a.resource),
                    f"<code>{_e(a.config_key)}</code>={_e(_fmt(a.config_value))}",
                    f"<code>{_e(a.checkpoint)}</code>" if a.checkpoint else "—",
                    _fmt(a.confidence, sig=2),
                )
                for a in rec.attributions
            ],
            numeric=(2, 6),
        )
    )
    P.append("<h3>因果链：八层信号</h3>")
    P.append(
        "<div class='legend'>"
        + "".join(f"<span><i style='background:{_LAYER_COLOR[l]}'></i>{_e(l)}</span>" for l in LAYERS)
        + "</div>"
    )
    P.append("<div class='panel'>")
    P.append(_chain_svg(result, list(rec.attributions)))
    P.append("</div>")

    # 五、第三方
    P.append("<h2>五、第三方校验：生成解释的人不给自己打分</h2>")
    P.append(
        f"<div class='note'><code>alpha_sigma = {_fmt(ver.alpha_sigma, sig=2)}</code> —— "
        "变化不到一个噪音尺度，与噪音不可区分。<br>"
        "只测 P 是不够的：一个「只要注了就都行」的判据也会全过，所以三个方向都要测。</div>"
    )
    P.append(
        _table(
            ("探针", "问的问题", "通过率", "个数"),
            [
                ("<code>P</code>", "去掉它，目标信号<b>要动</b>", _pct(ver.probe_pass_rate("P")), str(sum(1 for p in ver.probes if p.probe == "P"))),
                ("<code>F</code>", "去掉它，它声明<b>不影响</b>的信号必须不动", _pct(ver.probe_pass_rate("F")), str(sum(1 for p in ver.probes if p.probe == "F"))),
                ("<code>h</code>", "在别处注入等量<b>不同种类</b>的诱饵，目标信号必须不动", _pct(ver.probe_pass_rate("h")), str(sum(1 for p in ver.probes if p.probe == "h"))),
            ],
            numeric=(2, 3),
        )
    )
    P.append("<h3>反事实反向链</h3>")
    P.append(
        _table(
            ("根因", "去掉前", "去掉后", "同类剩", "链消失", "通过"),
            [
                (
                    f"<code>{_e(c.root_key)}</code>",
                    str(c.n_attributions_before),
                    str(c.n_attributions_after),
                    str(c.same_kind_after),
                    "是" if c.chain_disappeared else "否",
                    '<span class="ok">✓</span>' if c.passed else '<span class="bad">✗</span>',
                )
                for c in ver.counterfactuals
            ],
            numeric=(1, 2, 3),
        )
    )
    P.append("<h3>α 标定曲线（Δ/σ）</h3>")
    if ver.curves:
        fs = [f for f, _ in ver.curves[0].points]
        P.append(
            _table(
                ("根因", *[f"f={f:g}" for f in fs], "f*", "单调"),
                [
                    (
                        f"<code>{_e(c.kind)}</code>",
                        *[("inf" if math.isinf(d) else _fmt(d, sig=3)) for _f, d in c.points],
                        _fmt(c.f_star, sig=2) if c.f_star >= 0 else "—（开关式）",
                        "是" if c.monotone else "否",
                    )
                    for c in ver.curves
                ],
                numeric=tuple(range(1, len(fs) + 1)),
            )
        )
    P.append(
        "<div class='note'><b><code>nan_grad</code> 的曲线必然退化。</b> 它是开关式的："
        "梯度变成 NaN 这件事没有「幅度小一点」的中间态。一条测不出中间态的曲线不是缺陷，"
        "是它诚实报告了「这个原因不可调幅」。</div>"
    )
    P.append("<h3>对齐消融</h3>")
    P.append(
        _table(
            ("允许的假设", "救回归因", "召回", "说明"),
            [
                (f"<code>{_e(a.label)}</code>", f"{a.n_roots_recovered}/{a.n_roots_total}", _pct(a.recall), _e(a.error[:80]) if a.error else "—")
                for a in ver.ablations
            ],
            numeric=(1, 2),
        )
    )

    # 六、期望
    P.append("<h2>六、设计期望 vs 实测</h2>")
    P.append(
        "<div class='note'><b>允许期望被推翻。</b> 被推翻的那条最值钱 —— 它说明设计里有一句话是错的。</div>"
    )
    P.append(
        _table(
            ("期望", "声明", "实测", "判定", "依据"),
            [
                (
                    f"<code>{_e(r['key'])}</code>",
                    "是" if r["declared"] else "否",
                    "成立" if r["observed"] else "不成立",
                    '<span class="ok">✓ 兑现</span>' if r["held"] else '<span class="bad">✗ 被推翻</span>',
                    _e(r["note"]),
                )
                for r in rows
            ]
        )
    )

    # 七、验收
    P.append("<h2>七、验收</h2>")
    P.append(
        _table(
            ("验收项", "实测", "门槛", "关系", "通过"),
            [
                (
                    f"<code>{_e(a['check'])}</code>",
                    _fmt(a["value"], sig=5),
                    _fmt(a["threshold"], sig=5),
                    _e(a["op"]),
                    '<span class="ok">✓</span>' if a["passed"] else '<span class="bad">✗</span>',
                )
                for a in acc
            ],
            numeric=(1, 2),
        )
    )
    P.append(f"<div class='note'><b>{n_acc}/{len(acc)} 通过。</b></div>")

    # 八、血缘
    P.append("<h2>八、血缘</h2>")
    lin = result.lineage
    kinds: dict[str, int] = {}
    for n in lin.nodes:
        kinds[n.kind] = kinds.get(n.kind, 0) + 1
    P.append(
        f"<div class='note'>{len(lin.nodes)} 个节点 / {len(lin.edges)} 条边。两条不变量："
        "① <code>attrib</code> 的父节点恰好是它 <code>covered_anomalies</code> 里那几个检出；"
        "② <code>hop</code> 的父节点恰好是链上前一跳 —— 从任意一跳往上走，走到根就是第一因。</div>"
    )
    P.append(
        _table(
            ("节点类型", "个数"),
            [(f"<code>{_e(k)}</code>", str(v)) for k, v in sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))],
            numeric=(1,),
        )
    )

    P.append(
        "<div class='sub' style='margin-top:40px'>"
        "本页零外部资源：没有 CDN、没有字体、没有图片、没有 fetch。</div>"
    )
    P.append("</div></body></html>")
    return "\n".join(P)


def _rate(result: Any, which: str) -> float:
    from .report import _chain_hits, _five_answer_hits, _matched

    n = max(1, len(_matched(result)))
    return (_five_answer_hits(result) if which == "five" else _chain_hits(result)) / n
