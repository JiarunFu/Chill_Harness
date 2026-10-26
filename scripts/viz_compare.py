#!/usr/bin/env python3
"""
Visualize base vs CF job comparison.

Usage:
  python scripts/viz_compare.py <base_job_dir> <cf_job_dir> [--out out.png]
"""
from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import numpy as np

# ── palette ──────────────────────────────────────────────────────────────────
BG       = "#0f0f1a"
PANEL_BG = "#16213e"
BASE_C   = "#4fc3f7"   # light-blue  → BASE
CF_C     = "#ffb74d"   # amber       → CF
CF_OVH_C = "#ff8a65"   # deep-orange → CF overhead
PASS_C   = "#69f0ae"   # bright green
FAIL_C   = "#ff5252"   # bright red
ERR_C    = "#78909c"   # blue-grey
UP_C     = "#b9f6ca"   # rescued flip
DOWN_C   = "#ff6e6e"   # broke flip
SAME_C   = "#546e7a"   # unchanged

SUFFIX_RE = re.compile(r"__[A-Za-z0-9]{6,}$")
_CF_STEP_RE = re.compile(
    r'"changed_plan"\s*:\s*(true|false).*?'
    r'"planner_extra_llm_calls"\s*:\s*(\d+).*?'
    r'"planner_extra_input_tokens"\s*:\s*(\d+).*?'
    r'"planner_extra_output_tokens"\s*:\s*(\d+).*?'
    r'"planner_extra_cost_usd"\s*:\s*([\d.]+)',
    re.DOTALL,
)
_CF_MODE_RE      = re.compile(r'"planner_mode"\s*:\s*"([^"]+)"')
_CF_TRIGGERED_RE = re.compile(r'"planner_triggered"\s*:\s*(true|false)')
CF_LOG_RE = re.compile(
    r"\[KIRA CF STATS\]"
    r"\s+episodes_triggered=(\d+)"
    r"\s+changed_plans=(\d+)"
    r"\s+cf_llm_calls=(\d+)"
    r"\s+cf_input_tokens=(\d+)"
    r"\s+cf_output_tokens=(\d+)"
    r"\s+cf_cost_usd=([\d.]+)"
)

# ── data loading ──────────────────────────────────────────────────────────────

def _cf_from_trajectory(traj_path: Path) -> dict:
    """Parse CF planner overhead from per-step message text in trajectory.json."""
    empty = dict(llm_calls=0, input_tokens=0, output_tokens=0, cost_usd=0.0,
                 episodes_triggered=0, changed_plans=0, mode_counts={})
    try:
        data = json.load(open(traj_path))
    except (FileNotFoundError, json.JSONDecodeError):
        return empty

    llm_calls = inp = out = 0
    cost = 0.0
    triggered = changed = 0
    modes: dict[str, int] = {}

    for step in data.get("steps", []):
        if step.get("source") != "agent":
            continue
        msg = step.get("message", "")
        m = _CF_STEP_RE.search(msg)
        if not m:
            continue
        if m.group(1) == "true":
            changed += 1
        llm_calls += int(m.group(2))
        inp       += int(m.group(3))
        out       += int(m.group(4))
        cost      += float(m.group(5))

        tm = _CF_TRIGGERED_RE.search(msg)
        if tm and tm.group(1) == "true":
            triggered += 1

        mm = _CF_MODE_RE.search(msg)
        if mm:
            k = mm.group(1)
            modes[k] = modes.get(k, 0) + 1

    return dict(llm_calls=llm_calls, input_tokens=inp, output_tokens=out,
                cost_usd=cost, episodes_triggered=triggered, changed_plans=changed,
                mode_counts=modes)


def _cf_from_jsonl(trial_dir: Path) -> dict | None:
    """Aggregate CF planner stats from cf_planner_events.jsonl files (adaptive)."""
    agent_dir = trial_dir / "agent"
    if not agent_dir.exists():
        return None
    files = sorted(agent_dir.glob("episode-*/cf_planner_events.jsonl"))
    files += sorted(trial_dir.glob("cf_planner_events.jsonl"))
    if not files:
        return None
    llm_calls = inp = out = 0
    triggered = changed = 0
    modes: dict[str, int] = {}
    found = False
    for f in files:
        try:
            with f.open(errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    found = True
                    stats = rec.get("stats") or {}
                    llm_calls += int(stats.get("planner_extra_llm_calls", 0) or 0)
                    inp       += int(stats.get("planner_extra_input_tokens", 0) or 0)
                    out       += int(stats.get("planner_extra_output_tokens", 0) or 0)
                    if stats.get("planner_triggered"):
                        triggered += 1
                    if stats.get("changed_plan") or rec.get("changed"):
                        changed += 1
                    mode = stats.get("planner_mode") or rec.get("mode") or "unknown"
                    modes[mode] = modes.get(mode, 0) + 1
        except OSError:
            continue
    if not found:
        return None
    return dict(llm_calls=llm_calls, input_tokens=inp, output_tokens=out,
                cost_usd=0.0, episodes_triggered=triggered, changed_plans=changed,
                mode_counts=modes)


def _cf_from_log(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    last = None
    try:
        with log_path.open(errors="replace") as f:
            for line in f:
                m = CF_LOG_RE.search(line)
                if m:
                    last = m
    except OSError:
        return None
    if not last:
        return None
    return dict(episodes_triggered=int(last.group(1)), changed_plans=int(last.group(2)),
                llm_calls=int(last.group(3)), input_tokens=int(last.group(4)),
                output_tokens=int(last.group(5)), cost_usd=float(last.group(6)),
                mode_counts={})


def load_trials(job_dir: Path) -> dict[str, dict]:
    result_path = job_dir / "result.json"
    rewards: dict[str, float] = {}
    exceptions: set[str] = set()
    if result_path.exists():
        data = json.load(open(result_path))
        for ev in (data.get("stats", {}).get("evals", {}) or {}).values():
            for r_str, trials in ((ev.get("reward_stats") or {}).get("reward") or {}).items():
                for t in trials:
                    try: rewards[t] = float(r_str)
                    except ValueError: pass
            for _, trials in (ev.get("exception_stats") or {}).items():
                for t in trials:
                    exceptions.add(t)

    trials: dict[str, dict] = {}
    for child in sorted(job_dir.iterdir()):
        if not child.is_dir() or "__" not in child.name:
            continue
        task = SUFFIX_RE.sub("", child.name)
        traj = child / "agent" / "trajectory.json"

        fm: dict = {}
        if traj.exists():
            try:
                d = json.load(open(traj))
                fm = d.get("final_metrics", {}) or {}
            except json.JSONDecodeError:
                pass

        reward = rewards.get(child.name)
        is_err = child.name in exceptions or reward is None

        inp     = int(fm.get("total_prompt_tokens",    0) or 0)
        out     = int(fm.get("total_completion_tokens", 0) or 0)
        cached  = int(fm.get("total_cached_tokens",    0) or 0)
        cost    = float(fm.get("total_cost_usd",       0) or 0)

        cf = _cf_from_jsonl(child) or _cf_from_log(child / "trial.log") or _cf_from_trajectory(traj)

        trials[task] = dict(
            reward=reward, errored=is_err, passed=(reward is not None and reward >= 1.0),
            tokens=inp + out, input_tokens=inp, output_tokens=out,
            cached_tokens=cached, cost=cost,
            cf_input=cf["input_tokens"], cf_output=cf["output_tokens"],
            cf_cost=cf["cost_usd"], cf_llm_calls=cf["llm_calls"],
            cf_triggered=cf["episodes_triggered"], cf_changed=cf["changed_plans"],
            cf_modes=cf["mode_counts"],
        )
    return trials


def _status(t: dict) -> str:
    if t["errored"]: return "errored"
    if t["passed"]:  return "pass"
    return "fail"


def _mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    total = sum(math.comb(n, i) for i in range(n + 1) if i <= k or i >= n - k)
    return min(1.0, total / (2.0 ** n))


# ── axes styling ──────────────────────────────────────────────────────────────

def _style(ax):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_color("#333")
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.xaxis.label.set_color("#bbb")
    ax.yaxis.label.set_color("#bbb")
    ax.title.set_color("#ddd")


def _bar_label(ax, bars, fmt="{:.0f}"):
    for b in bars:
        h = b.get_height()
        if h > 0:
            ax.text(b.get_x() + b.get_width() / 2, h + ax.get_ylim()[1] * 0.01,
                    fmt.format(h), ha="center", va="bottom", color="white", fontsize=7)


# ── panels ────────────────────────────────────────────────────────────────────

def _panel_result_grid(ax, base, cf, common):
    """Per-task pass/fail/err heatmap with flip highlights."""
    n = len(common)
    status_num = {"pass": 1, "fail": -1, "errored": 0}
    grid = np.array([[status_num[_status(base[t])] for t in common],
                     [status_num[_status(cf[t])]   for t in common]], dtype=float)

    cmap = ListedColormap([FAIL_C, ERR_C, PASS_C])
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=-1, vmax=1, interpolation="none")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["BASE", "CF"], color="#ddd", fontsize=9, fontweight="bold")
    ax.set_xticks(range(n))
    labels = [t.replace("polyglot_", "").replace("_", "\n", 1) for t in common]
    ax.set_xticklabels(labels, rotation=90, fontsize=5, color="#aaa")

    # Draw BASE/CF row borders
    for y, color in [(0, BASE_C), (1, CF_C)]:
        ax.add_patch(plt.Rectangle((-0.5, y - 0.5), n, 1,
                                   fill=False, edgecolor=color, lw=1.2, zorder=4))

    # Highlight flip columns
    for i, t in enumerate(common):
        bs, cs = _status(base[t]), _status(cf[t])
        if bs != cs:
            ec = UP_C if cs == "pass" else DOWN_C
            ax.add_patch(plt.Rectangle((i - 0.5, -0.5), 1, 2,
                                       fill=False, edgecolor=ec, lw=2, zorder=5))

    leg = [mpatches.Patch(color=PASS_C, label="pass"),
           mpatches.Patch(color=FAIL_C, label="fail"),
           mpatches.Patch(color=ERR_C,  label="errored"),
           mpatches.Patch(color=UP_C,   label="rescued ↑"),
           mpatches.Patch(color=DOWN_C, label="broke ↓")]
    ax.legend(handles=leg, loc="upper right", fontsize=6.5,
              facecolor=BG, labelcolor="white", edgecolor="#444", ncol=5)
    ax.set_title("Per-task result grid  (blue border = BASE row, amber = CF row, "
                 "bright outline = flip)", fontsize=8)


def _panel_pass_counts(ax, base, cf):
    """Grouped bar: pass/fail/err, BASE=blue, CF=amber."""
    statuses = ["pass", "fail", "errored"]
    bc = [sum(1 for t in base.values() if _status(t) == s) for s in statuses]
    cc = [sum(1 for t in cf.values()   if _status(t) == s) for s in statuses]
    n_b, n_c = len(base), len(cf)

    x = np.arange(3)
    w = 0.38
    bars_b = ax.bar(x - w/2, bc, w, color=BASE_C, alpha=0.9, label=f"BASE (n={n_b})")
    bars_c = ax.bar(x + w/2, cc, w, color=CF_C,   alpha=0.9, label=f"CF   (n={n_c})")

    # pass-rate annotations
    for bars, counts, total in [(bars_b, bc, n_b), (bars_c, cc, n_c)]:
        for bar, cnt in zip(bars, counts):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                        f"{cnt}\n({cnt/total*100:.0f}%)", ha="center", va="bottom",
                        color="white", fontsize=6.5, linespacing=1.3)

    ax.set_xticks(x)
    ax.set_xticklabels(statuses)
    ax.set_ylabel("count")
    ax.set_title("Result counts  (BASE=blue / CF=amber)")
    ax.legend(facecolor=BG, labelcolor="white", edgecolor="#444", fontsize=7)


def _panel_cost(ax, base, cf):
    """Stacked cost bar: BASE main | CF main + CF overhead."""
    b_main = sum(t["cost"] for t in base.values())
    c_main = sum(t["cost"] for t in cf.values())
    c_ovh  = sum(t["cf_cost"] for t in cf.values())

    b_tok  = sum(t["tokens"] for t in base.values())
    c_tok  = sum(t["tokens"] for t in cf.values())
    c_ovh_tok = sum(t["cf_input"] + t["cf_output"] for t in cf.values())
    c_ovh_ratio = c_ovh_tok / c_tok * 100 if c_tok else 0

    labels = ["BASE", "CF"]
    mains  = [b_main, c_main]
    ovhs   = [0,      c_ovh]

    x = np.arange(2)
    w = 0.45
    b1 = ax.bar(x, mains, w, color=[BASE_C, CF_C], alpha=0.9, label="main LLM")
    b2 = ax.bar(x, ovhs,  w, bottom=mains, color=[BASE_C, CF_OVH_C], alpha=0.75,
                label="CF overhead")

    for bar, main, ovh, tok, otok in zip(x, mains, ovhs, [b_tok, c_tok], [0, c_ovh_tok]):
        ax.text(bar, main + ovh + ax.get_ylim()[1] * 0.01,
                f"${main+ovh:.2f}\n{(tok+otok)/1e6:.1f}M tok",
                ha="center", va="bottom", color="white", fontsize=7, linespacing=1.3)

    ax.text(1, c_main + c_ovh / 2,
            f"+{c_ovh_ratio:.1f}%\noverhead",
            ha="center", va="center", color="white", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#333", edgecolor="none"))

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("cost (USD)")
    ax.set_title("Cost breakdown  (CF overhead = planner LLM calls)")
    ax.legend(facecolor=BG, labelcolor="white", edgecolor="#444", fontsize=7)


def _panel_token_scatter(ax, base, cf, common):
    """Token scatter per task, colored by flip outcome."""
    flip_map = {
        ("pass",    "pass"):    (SAME_C,  "same→pass"),
        ("fail",    "fail"):    (SAME_C,  "same→fail"),
        ("errored", "errored"): (SAME_C,  "same→err"),
        ("pass",    "fail"):    (DOWN_C,  "broke ↓"),
        ("pass",    "errored"): (DOWN_C,  "broke ↓"),
        ("fail",    "pass"):    (UP_C,    "rescued ↑"),
        ("errored", "pass"):    (UP_C,    "rescued ↑"),
        ("fail",    "errored"): ("#f39c12","changed"),
        ("errored", "fail"):    ("#f39c12","changed"),
    }
    seen_labels: set[str] = set()
    for t in common:
        bt, ct = base[t]["tokens"], cf[t]["tokens"]
        key = (_status(base[t]), _status(cf[t]))
        color, label = flip_map.get(key, (SAME_C, "other"))
        kw = dict(label=label) if label not in seen_labels else {}
        seen_labels.add(label)
        ax.scatter(max(bt, 1), max(ct, 1), color=color, s=70, alpha=0.85,
                   edgecolors="white", linewidths=0.35, zorder=3, **kw)

    all_toks = [t["tokens"] for t in {**base, **cf}.values() if t["tokens"] > 0]
    lo, hi = min(all_toks, default=100), max(all_toks, default=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot([lo, hi], [lo, hi], "--", color="#555", lw=0.8, zorder=1)
    ax.set_xlabel("BASE tokens"); ax.set_ylabel("CF tokens")
    ax.set_title("Token per task  (above diagonal = CF used more)")
    ax.legend(fontsize=6.5, facecolor=BG, labelcolor="white",
              edgecolor="#444", markerscale=0.9)


def _panel_mcnemar(ax, base, cf, common):
    """McNemar 2×2 contingency table with p-value."""
    pp = pf = fp = ff = 0
    for t in common:
        br, cr = base[t]["reward"], cf[t]["reward"]
        if br is None or cr is None:
            continue
        bp, cp = br >= 1.0, cr >= 1.0
        if   bp and cp:      pp += 1
        elif bp and not cp:  pf += 1
        elif not bp and cp:  fp += 1
        else:                ff += 1

    p_val = _mcnemar_p(pf, fp)
    sig   = p_val < 0.05

    table = np.array([[pp, pf], [fp, ff]], dtype=float)
    total = table.sum()
    norm  = table / (total or 1)

    cmap_heat = plt.cm.Blues
    ax.imshow(norm, cmap=cmap_heat, vmin=0, vmax=norm.max() * 1.2, aspect="auto")

    for i in range(2):
        for j in range(2):
            val = int(table[i, j])
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if norm[i, j] > norm.max() * 0.4 else "#ccc",
                    fontsize=14, fontweight="bold")

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["CF pass", "CF fail"], color="#ddd")
    ax.set_yticklabels(["BASE pass", "BASE fail"], color="#ddd")

    sig_color = "#69f0ae" if sig else "#ff8a65"
    sig_text  = f"SIGNIFICANT ✓" if sig else "not significant"
    ax.set_title(
        f"McNemar 2×2   p={p_val:.4f}\n"
        f"flips ↑={fp}  ↓={pf}  net={fp-pf:+d}",
        fontsize=8, color="#ddd",
    )
    ax.text(0.5, -0.22, sig_text, transform=ax.transAxes,
            ha="center", color=sig_color, fontsize=9, fontweight="bold")


def _panel_cf_modes(ax, cf):
    """CF planner mode distribution (skip vs full, etc.) across all trials."""
    totals: dict[str, int] = {}
    for t in cf.values():
        for mode, n in t["cf_modes"].items():
            totals[mode] = totals.get(mode, 0) + n

    if not totals:
        ax.text(0.5, 0.5, "no CF mode data", transform=ax.transAxes,
                ha="center", va="center", color="#888")
        ax.set_title("CF planner mode distribution")
        return

    modes  = sorted(totals, key=totals.get, reverse=True)
    counts = [totals[m] for m in modes]
    colors = [CF_OVH_C if m == "full" else SAME_C for m in modes]

    bars = ax.barh(range(len(modes)), counts, color=colors, alpha=0.9)
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels(modes, color="#ccc")
    ax.set_xlabel("episode count")
    ax.set_title("CF planner mode distribution")

    total_ep = sum(counts)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + total_ep * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{cnt}  ({cnt/total_ep*100:.1f}%)", va="center", color="#ccc", fontsize=7)

    ax.text(0.97, 0.97,
            f"CF triggered: {sum(t['cf_triggered'] for t in cf.values())}\n"
            f"plans changed: {sum(t['cf_changed'] for t in cf.values())}",
            transform=ax.transAxes, ha="right", va="top",
            color="#aaa", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d0d23", edgecolor="#444"))


def _panel_per_task_cost(ax, base, cf, common):
    """Horizontal bar: per-task cost side-by-side, sorted by base cost."""
    order = sorted(common, key=lambda t: base[t]["cost"], reverse=True)
    y     = np.arange(len(order))
    h     = 0.38

    b_costs = [base[t]["cost"] for t in order]
    c_main  = [cf[t]["cost"]   for t in order]
    c_ovh   = [cf[t]["cf_cost"] for t in order]

    ax.barh(y + h/2, b_costs, h, color=BASE_C,    alpha=0.9, label="BASE main")
    ax.barh(y - h/2, c_main,  h, color=CF_C,      alpha=0.9, label="CF main")
    ax.barh(y - h/2, c_ovh,   h, left=c_main,     color=CF_OVH_C, alpha=0.75, label="CF overhead")

    short = [t.replace("polyglot_", "") for t in order]
    ax.set_yticks(y)
    ax.set_yticklabels(short, fontsize=5.5, color="#bbb")
    ax.set_xlabel("cost (USD)")
    ax.set_title("Per-task cost  (BASE=blue / CF main=amber / CF overhead=orange)")
    ax.legend(facecolor=BG, labelcolor="white", edgecolor="#444", fontsize=6.5)


# ── main figure ───────────────────────────────────────────────────────────────

def make_figure(base: dict, cf: dict, base_label: str, cf_label: str) -> plt.Figure:
    common = sorted(set(base) & set(cf))

    fig = plt.figure(figsize=(22, 20))
    fig.patch.set_facecolor(BG)

    # Layout: 4 rows
    # Row 0 (tall): result grid (full width)
    # Row 1: pass counts | cost breakdown
    # Row 2: token scatter | mcnemar
    # Row 3 (tall): per-task cost (full width)
    # Row 4: CF planner modes (left half)
    outer = gridspec.GridSpec(5, 2, figure=fig,
                              hspace=0.52, wspace=0.3,
                              left=0.07, right=0.97, top=0.93, bottom=0.04,
                              height_ratios=[2, 1.8, 1.8, 2.5, 1.8])

    ax_grid     = fig.add_subplot(outer[0, :])
    ax_counts   = fig.add_subplot(outer[1, 0])
    ax_cost     = fig.add_subplot(outer[1, 1])
    ax_scatter  = fig.add_subplot(outer[2, 0])
    ax_mcnemar  = fig.add_subplot(outer[2, 1])
    ax_pertask  = fig.add_subplot(outer[3, :])
    ax_modes    = fig.add_subplot(outer[4, 0])
    ax_blank    = fig.add_subplot(outer[4, 1])

    all_axes = [ax_grid, ax_counts, ax_cost, ax_scatter,
                ax_mcnemar, ax_pertask, ax_modes, ax_blank]
    for ax in all_axes:
        _style(ax)
    ax_blank.set_visible(False)

    _panel_result_grid(ax_grid,   base, cf, common)
    _panel_pass_counts(ax_counts, base, cf)
    _panel_cost(ax_cost,          base, cf)
    _panel_token_scatter(ax_scatter, base, cf, common)
    _panel_mcnemar(ax_mcnemar,    base, cf, common)
    _panel_per_task_cost(ax_pertask, base, cf, common)
    _panel_cf_modes(ax_modes,     cf)

    fig.suptitle(
        f"BASE vs CF  ·  {base_label}\n        vs  {cf_label}",
        color="#eee", fontsize=10, y=0.97, ha="left", x=0.07,
    )
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base_job_dir", type=Path)
    p.add_argument("cf_job_dir",   type=Path)
    p.add_argument("--out", type=Path, default=Path("compare.png"))
    args = p.parse_args()

    base = load_trials(args.base_job_dir)
    cf   = load_trials(args.cf_job_dir)
    fig  = make_figure(base, cf, args.base_job_dir.name, args.cf_job_dir.name)
    fig.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
