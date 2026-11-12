"""Command line entry points.

    python -m rca_agent.cli analyze EV-0007      # dossier for one event
    python -m rca_agent.cli evaluate             # accuracy over all labeled events
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .graph import build_graph, default_llm

DATA = Path(__file__).resolve().parents[2] / "data"


def evaluate(graph=None) -> dict:
    graph = graph or build_graph(llm=None)  # evaluation scores the rules, not the prose
    events = json.loads((DATA / "events.json").read_text())
    hits, confusion = 0, Counter()
    for ev in events:
        out = graph.invoke({"event_id": ev["event_id"]})
        pred = out["hypotheses"][0]["cause"]
        hits += pred == ev["true_cause"]
        confusion[(ev["true_cause"], pred)] += 1
    return {"events": len(events), "top1_accuracy": round(hits / len(events), 3),
            "errors": {f"{t} -> {p}": n for (t, p), n in confusion.items() if t != p}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("event_id")
    a.add_argument("--json", action="store_true", help="print the full state instead of the report")
    sub.add_parser("evaluate")
    args = ap.parse_args(argv)

    if args.cmd == "analyze":
        out = build_graph(default_llm()).invoke({"event_id": args.event_id})
        print(json.dumps(out, indent=1, default=str) if args.json else out["report"])
    else:
        print(json.dumps(evaluate(), indent=1))


if __name__ == "__main__":
    main()
