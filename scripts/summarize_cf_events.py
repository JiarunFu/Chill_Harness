#!/usr/bin/env python3
"""Small helper for summarizing cf_planner_events.jsonl files."""

import json
import sys
from collections import Counter
from pathlib import Path


def main(path: str) -> None:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    completion_gate_rows = [r for r in rows if r.get("event_type") == "completion_gate"]
    stats = [r.get("stats", {}) for r in rows if r.get("event_type") != "completion_gate"]
    print(f"events={len(rows)}")
    print(f"planner_events={len(stats)}")
    print(f"completion_gate_events={len(completion_gate_rows)}")
    print("mode_histogram", Counter(s.get("planner_mode", "unknown") for s in stats))
    print("triggered", sum(1 for s in stats if s.get("planner_triggered")))
    print("changed", sum(1 for s in stats if s.get("changed_plan")))
    print("would_change", sum(1 for s in stats if s.get("would_change_plan")))
    print("extra_llm_calls", sum(int(s.get("planner_extra_llm_calls", 0) or 0) for s in stats))
    print("extra_input_tokens", sum(int(s.get("planner_extra_input_tokens", 0) or 0) for s in stats))
    print("extra_output_tokens", sum(int(s.get("planner_extra_output_tokens", 0) or 0) for s in stats))
    print("avg_latency_ms", (sum(float(s.get("planner_latency_ms", 0.0) or 0.0) for s in stats) / max(1, len(stats))))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: summarize_cf_events.py path/to/cf_planner_events.jsonl")
        raise SystemExit(2)
    main(sys.argv[1])
