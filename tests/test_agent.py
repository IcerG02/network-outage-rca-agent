import json
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rca_agent.graph import build_graph, route_after_spatial
from rca_agent.tools import get_event, search_procedures, spatial_pattern

EVENTS = json.loads((Path(__file__).resolve().parents[1] / "data" / "events.json").read_text())


def first_event(cause):
    return next(e["event_id"] for e in EVENTS if e["true_cause"] == cause)


def test_label_is_hidden_from_the_agent():
    assert "true_cause" not in get_event.invoke({"event_id": EVENTS[0]["event_id"]})


def test_spatial_scopes():
    assert spatial_pattern.invoke({"event_id": first_event("fiber_cut")})["scope"] == "pon"
    assert spatial_pattern.invoke({"event_id": first_event("equipment_failure")})["scope"] == "olt"
    assert spatial_pattern.invoke({"event_id": first_event("weather")})["scope"] == "region"


def test_single_port_outages_skip_the_weather_lookup():
    assert route_after_spatial({"spatial": {"scope": "pon"}}) == "news"
    out = build_graph().invoke({"event_id": first_event("fiber_cut")})
    assert out.get("weather") is None


def test_retrieval_finds_the_right_procedure():
    top = search_procedures.invoke({"query": "truck hit a utility pole", "k": 1})[0]
    assert top["doc"] == "pole_impact"


def test_probabilities_sum_to_one_and_cite_evidence():
    out = build_graph().invoke({"event_id": first_event("weather")})
    assert abs(sum(h["probability"] for h in out["hypotheses"]) - 1) < 0.01
    assert out["hypotheses"][0]["evidence"]


def test_llm_writes_only_the_report():
    fake = FakeListChatModel(responses=["# Dossier\nMost likely: severe weather."])
    out = build_graph(llm=fake).invoke({"event_id": first_event("weather")})
    assert out["report"].startswith("# Dossier")
    assert out["hypotheses"][0]["cause"] == "weather"  # scoring is independent of the LLM
