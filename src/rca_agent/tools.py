"""Tools the agent uses to gather evidence.

Each tool is a LangChain ``@tool`` so it can also be bound to a tool-calling
LLM; the default graph calls them directly in a fixed order. Data sources are
local JSON files here, behind small functions that a real deployment would
swap for the network database, a weather API (e.g. Open-Meteo) and a news feed.
"""
from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path(__file__).resolve().parents[2] / "data"


@lru_cache
def _load(name: str):
    return json.loads((DATA / name).read_text())


@lru_cache
def _topology() -> pd.DataFrame:
    return pd.read_csv(DATA / "topology.csv").set_index("olt_id")


@tool
def get_event(event_id: str) -> dict:
    """Return an outage event (time window and affected OLTs/ports), without its label."""
    ev = next(e for e in _load("events.json") if e["event_id"] == event_id)
    return {k: v for k, v in ev.items() if k != "true_cause"}


@tool
def spatial_pattern(event_id: str) -> dict:
    """Classify how far an outage spread: one PON port, a whole OLT, or several OLTs in a region."""
    ev = get_event.invoke({"event_id": event_id})
    topo = _topology()
    olts = [a["olt_id"] for a in ev["affected"]]
    ports = sum(len(a["pon_ports"]) for a in ev["affected"])
    per_port = int(topo.loc[olts[0], "onus_per_port"])
    if len(olts) > 1:
        scope = "region"
    elif len(ev["affected"][0]["pon_ports"]) >= int(topo.loc[olts[0], "pon_ports"]):
        scope = "olt"
    else:
        scope = "pon"
    first = topo.loc[olts[0]]
    return {"scope": scope, "olts": olts, "ports_down": ports, "onus_down": ports * per_port,
            "cities": sorted({topo.loc[o, "city"] for o in olts}), "region": first["region"],
            "lat": float(first["lat"]), "lon": float(first["lon"])}


@tool
def weather_at(city: str, when: str) -> dict:
    """Weather for a city in the hour of an event: precipitation, wind gusts and lightning."""
    hour = datetime.fromisoformat(when).replace(minute=0, second=0).isoformat()
    for w in _load("weather.json"):
        if w["city"] == city and w["hour"] == hour:
            return w
    return {"city": city, "hour": hour, "precip_mm": 0.0, "gust_kmh": 10.0, "lightning_strikes": 0}


@tool
def local_news(city: str, date: str) -> list[str]:
    """Local news headlines for a city on a given date (YYYY-MM-DD)."""
    return [n["headline"] for n in _load("news.json") if n["city"] == city and n["date"] == date]


@lru_cache
def _procedure_index():
    chunks = []
    for doc in sorted((DATA / "procedures").glob("*.md")):
        for para in doc.read_text(encoding="utf-8").split("\n\n"):
            if para.strip() and not para.startswith("#"):
                chunks.append((doc.stem, para.strip()))
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform([c[1] for c in chunks])
    return chunks, vec, matrix


@tool
def search_procedures(query: str, k: int = 2) -> list[dict]:
    """Retrieve the most relevant passages from internal operating procedures (RAG)."""
    chunks, vec, matrix = _procedure_index()
    scores = cosine_similarity(vec.transform([query]), matrix)[0]
    top = scores.argsort()[::-1][:k]
    return [{"doc": chunks[i][0], "text": chunks[i][1], "score": round(float(scores[i]), 3)} for i in top]
