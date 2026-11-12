"""LangGraph workflow that builds a root-cause dossier for one outage.

    load_event -> spatial -> (weather, if spread beyond one port) -> news
               -> procedures (RAG) -> score -> write_report

Most of the work is deterministic: rules and data lookups produce the evidence
and the scores. The LLM only writes the final narrative, and only from the
evidence it is given, which keeps cost low and every claim traceable. Without
an LLM the report falls back to a template, so the graph runs offline.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .tools import get_event, local_news, search_procedures, spatial_pattern, weather_at

LABELS = {
    "fiber_cut": "Fiber cut or splitter failure",
    "vehicle_pole": "Vehicle impact on a pole",
    "power_outage": "Power outage at the site",
    "equipment_failure": "OLT equipment failure",
    "weather": "Severe weather",
}
KEYWORDS = {
    "vehicle_pole": ("pole", "truck", "vehicle", "car", "bus"),
    "power_outage": ("power cut", "electricity", "blackout", "power outage"),
    "weather": ("storm", "wind", "tree", "lightning", "rain"),
}


class RCAState(TypedDict, total=False):
    event_id: str
    event: dict
    spatial: dict
    weather: dict | None
    news: list[str]
    procedures: list[dict]
    hypotheses: list[dict]
    report: str


def load_event(s: RCAState) -> RCAState:
    return {"event": get_event.invoke({"event_id": s["event_id"]})}


def spatial(s: RCAState) -> RCAState:
    return {"spatial": spatial_pattern.invoke({"event_id": s["event_id"]})}


def route_after_spatial(s: RCAState) -> str:
    # A single dead port is a local physical fault; weather only matters when
    # the outage spread to a whole OLT or a region.
    return "news" if s["spatial"]["scope"] == "pon" else "weather"


def weather(s: RCAState) -> RCAState:
    # Check every affected city and keep the worst conditions.
    obs = [weather_at.invoke({"city": c, "when": s["event"]["start"]}) for c in s["spatial"]["cities"]]
    worst = max(obs, key=lambda w: (w["gust_kmh"] >= 60 or w["lightning_strikes"] >= 20, w["gust_kmh"]))
    return {"weather": worst}


def news(s: RCAState) -> RCAState:
    date = s["event"]["start"][:10]
    heads = []
    for city in s["spatial"]["cities"]:
        heads += local_news.invoke({"city": city, "date": date})
    return {"news": heads}


def procedures(s: RCAState) -> RCAState:
    sp = s["spatial"]
    query = f"{sp['scope']} outage {'one PON port' if sp['scope'] == 'pon' else 'whole OLT'} " + " ".join(s.get("news", []))
    return {"procedures": search_procedures.invoke({"query": query, "k": 2})}


def _news_hits(news: list[str], cause: str) -> list[str]:
    return [h for h in news if any(k in h.lower() for k in KEYWORDS.get(cause, ()))]


def score(s: RCAState) -> RCAState:
    """Rule-based scoring; every point added is recorded as evidence."""
    sp, w, news_ = s["spatial"], s.get("weather"), s.get("news", [])
    severe = bool(w) and (w["gust_kmh"] >= 60 or w["lightning_strikes"] >= 20 or w["precip_mm"] >= 15)
    raw: dict[str, tuple[float, list[str]]] = {c: (0.05, []) for c in LABELS}

    def add(cause: str, pts: float, why: str) -> None:
        val, ev = raw[cause]
        raw[cause] = (val + pts, ev + [why])

    scope_txt = {"pon": "only one PON port down", "olt": "every port of one OLT down",
                 "region": f"{len(sp['olts'])} OLTs down in the {sp['region']} region"}[sp["scope"]]
    if sp["scope"] == "pon":
        add("fiber_cut", 0.6, scope_txt)
        add("vehicle_pole", 0.25, scope_txt)
    elif sp["scope"] == "olt":
        add("power_outage", 0.4, scope_txt)
        add("equipment_failure", 0.4, scope_txt)
    else:
        add("weather", 0.5, scope_txt)
        add("power_outage", 0.2, scope_txt)

    if severe:
        add("weather", 0.6, f"severe weather in {w['city']}: gusts {w['gust_kmh']} km/h, "
                            f"{w['precip_mm']} mm rain, {w['lightning_strikes']} lightning strikes")
        add("power_outage", 0.1, "storms often cause power cuts")
    elif w:
        add("equipment_failure", 0.15, "calm weather in the outage window")
        add("weather", -0.4, "calm weather in the outage window")

    for cause in KEYWORDS:
        for h in _news_hits(news_, cause):
            add(cause, 0.7, f'news: "{h}"')
    if sp["scope"] == "olt" and not severe and not _news_hits(news_, "power_outage"):
        add("equipment_failure", 0.2, "no weather or power-cut evidence")

    total = sum(max(v, 0) for v, _ in raw.values())
    hyps = [{"cause": c, "label": LABELS[c], "probability": round(max(v, 0) / total, 3), "evidence": ev}
            for c, (v, ev) in raw.items()]
    hyps.sort(key=lambda h: h["probability"], reverse=True)
    return {"hypotheses": hyps}


def _template_report(s: RCAState) -> str:
    sp, ev = s["spatial"], s["event"]
    start = datetime.fromisoformat(ev["start"])
    mins = int((datetime.fromisoformat(ev["end"]) - start).total_seconds() // 60)
    lines = [f"# Outage {s['event_id']}",
             f"{start:%Y-%m-%d %H:%M}, {mins} min · {', '.join(sp['cities'])} · "
             f"{sp['onus_down']:,} ONUs on {sp['ports_down']} ports ({sp['scope']} scope)", "",
             "## Probable causes"]
    for h in s["hypotheses"][:3]:
        lines.append(f"- **{h['label']}** ({h['probability']:.0%})")
        lines += [f"  - {e}" for e in h["evidence"]]
    lines += ["", "## Recommended procedure"]
    lines += [f"- [{p['doc']}] {p['text']}" for p in s["procedures"][:1]]
    return "\n".join(lines)


PROMPT = """You are a network operations analyst. Write a short root-cause dossier
in Markdown for the outage below, using ONLY the evidence given. State the most
likely cause with its probability, the evidence for it, what would confirm it,
and the first action from the procedure. Do not invent facts.

Evidence (JSON):
{evidence}"""


def make_writer(llm: Any | None = None):
    """Return the report node. With an LLM it writes prose; otherwise a template."""
    def write_report(s: RCAState) -> RCAState:
        if llm is None:
            return {"report": _template_report(s)}
        evidence = {k: s.get(k) for k in ("event", "spatial", "weather", "news", "hypotheses", "procedures")}
        msg = llm.invoke(PROMPT.format(evidence=json.dumps(evidence, indent=1, default=str)))
        return {"report": getattr(msg, "content", str(msg))}
    return write_report


def default_llm():
    """Anthropic model if ANTHROPIC_API_KEY is set, else None (template report)."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=os.getenv("RCA_MODEL", "claude-sonnet-5-5"), temperature=0, max_tokens=800)


def build_graph(llm: Any | None = None):
    g = StateGraph(RCAState)
    for name, fn in [("load_event", load_event), ("spatial", spatial), ("weather", weather), ("news", news),
                     ("procedures", procedures), ("score", score), ("write_report", make_writer(llm))]:
        g.add_node(name, fn)
    g.add_edge(START, "load_event")
    g.add_edge("load_event", "spatial")
    g.add_conditional_edges("spatial", route_after_spatial, {"weather": "weather", "news": "news"})
    g.add_edge("weather", "news")
    g.add_edge("news", "procedures")
    g.add_edge("procedures", "score")
    g.add_edge("score", "write_report")
    g.add_edge("write_report", END)
    return g.compile()
