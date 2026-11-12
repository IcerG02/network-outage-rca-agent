"""Generate a fictitious network, outage events and external context.

    python -m rca_agent.synthetic        # writes data/*.json, data/topology.csv

Every event has a hidden ``true_cause`` used only by ``rca_agent.evaluate``;
the agent never sees it.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
REGIONS = {
    "North": [("Alpha City", -22.90, -47.06), ("Beta Town", -22.75, -47.15), ("Gamma Ville", -22.82, -46.95)],
    "South": [("Delta Port", -23.55, -46.63), ("Epsilon Bay", -23.67, -46.54), ("Zeta Falls", -23.48, -46.85)],
}
CAUSES = ["fiber_cut", "vehicle_pole", "power_outage", "equipment_failure", "weather"]


def build(seed: int = 42, n_events: int = 40) -> None:
    rnd = random.Random(seed)
    DATA.mkdir(exist_ok=True)

    topo = []
    for region, cities in REGIONS.items():
        for city, lat, lon in cities:
            for k in range(3):
                topo.append({"olt_id": f"OLT-{len(topo) + 1:03d}", "city": city, "region": region,
                             "lat": round(lat + rnd.uniform(-.03, .03), 4),
                             "lon": round(lon + rnd.uniform(-.03, .03), 4),
                             "pon_ports": 16, "onus_per_port": 48})
    pd.DataFrame(topo).to_csv(DATA / "topology.csv", index=False)
    by_region = {r: [t for t in topo if t["region"] == r] for r in REGIONS}

    events, weather, news = [], [], []
    t0 = datetime(2026, 9, 1)
    for i in range(n_events):
        cause = CAUSES[i % len(CAUSES)]
        start = t0 + timedelta(hours=rnd.randint(0, 24 * 60), minutes=rnd.choice([0, 15, 30, 45]))
        olt = rnd.choice(topo)
        if cause in ("fiber_cut", "vehicle_pole"):
            affected = [{"olt_id": olt["olt_id"], "pon_ports": [rnd.randint(1, 16)]}]
        elif cause in ("power_outage", "equipment_failure"):
            affected = [{"olt_id": olt["olt_id"], "pon_ports": list(range(1, 17))}]
        else:  # weather hits several OLTs of a region, partially
            olts = rnd.sample(by_region[olt["region"]], k=rnd.randint(3, 6))
            affected = [{"olt_id": o["olt_id"], "pon_ports": sorted(rnd.sample(range(1, 17), rnd.randint(4, 16)))}
                        for o in olts]
        duration = rnd.randint(25, 240)
        events.append({"event_id": f"EV-{i + 1:04d}", "start": start.isoformat(),
                       "end": (start + timedelta(minutes=duration)).isoformat(),
                       "affected": affected, "true_cause": cause})

        city = olt["city"]
        severe = cause == "weather" or (cause == "power_outage" and rnd.random() < .3)
        weather.append({"city": city, "hour": start.replace(minute=0).isoformat(),
                        "precip_mm": round(rnd.uniform(15, 60) if severe else rnd.uniform(0, 3), 1),
                        "gust_kmh": round(rnd.uniform(65, 110) if severe else rnd.uniform(5, 35), 1),
                        "lightning_strikes": rnd.randint(20, 300) if severe else rnd.randint(0, 2)})
        # news coverage is incomplete, as in real life
        if cause == "vehicle_pole" and rnd.random() < .7:
            news.append({"city": city, "date": start.date().isoformat(),
                         "headline": f"Truck hits utility pole on a main avenue in {city}; traffic blocked"})
        elif cause == "power_outage" and rnd.random() < .6:
            news.append({"city": city, "date": start.date().isoformat(),
                         "headline": f"Power cut leaves neighborhoods of {city} without electricity"})
        elif cause == "weather" and rnd.random() < .7:
            news.append({"city": city, "date": start.date().isoformat(),
                         "headline": f"Storm with strong winds topples trees across {city}"})
        if rnd.random() < .5:  # noise: irrelevant local news
            news.append({"city": city, "date": start.date().isoformat(),
                         "headline": f"{city} city hall announces new park renovation"})

    (DATA / "events.json").write_text(json.dumps(events, indent=1))
    (DATA / "weather.json").write_text(json.dumps(weather, indent=1))
    (DATA / "news.json").write_text(json.dumps(news, indent=1))
    print(f"{len(topo)} OLTs, {len(events)} events -> {DATA}")


if __name__ == "__main__":
    build()
