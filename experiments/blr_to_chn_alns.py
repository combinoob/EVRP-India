import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import polyline  # pip install polyline
from dotenv import load_dotenv
import os
import heapq


# =========================
# Load ENV
# =========================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY") or ""
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/recommend")
EV_CHARGING_CSV = os.getenv("EV_CHARGING_CSV", "")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY. Put it in evrp_india/.env or set env var.")
if not EV_CHARGING_CSV:
    raise ValueError("Missing EV_CHARGING_CSV. Put absolute path in evrp_india/.env or set env var.")


# =========================
# Trip config
# =========================
ORIGIN = (12.9610, 77.6100)      # Bengaluru
DESTINATION = (13.0827, 80.2707) # Chennai
DEPARTURE_TIME = datetime.now().replace(microsecond=0) + timedelta(minutes=2)

# Vehicle / baseline energy
BATTERY_KWH = 60.0
START_SOC = 0.55
MIN_SOC = 0.10
TARGET_SOC_AT_CHARGE = 0.80
WH_PER_KM = 170.0
CHARGER_KW_ASSUMED = 50.0

# Availability penalty (avoid low-prob chargers)
LAMBDA_RISK = 8.0

# Backend recommend controls
RADIUS_KM = 50      # for highways, >10 is safer
K_RECOMMEND = 10
LAMBDA_DECAY = 0.05

# ALNS
ALNS_ITERS = 250
DESTROY_MIN = 1
DESTROY_MAX = 3
SEED = 42


random.seed(SEED)
np.random.seed(SEED)


# =========================
# Helpers
# =========================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def soc_to_kwh(soc: float) -> float:
    return soc * BATTERY_KWH

def kwh_to_soc(kwh: float) -> float:
    return kwh / BATTERY_KWH

def to_google_latlng(p: Tuple[float, float]) -> str:
    return f"{p[0]},{p[1]}"

def dt_to_unix_seconds(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

def fmt_backend_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")

def leg_energy_kwh(distance_m: float) -> float:
    km = distance_m / 1000.0
    return (WH_PER_KM * km) / 1000.0

def charge_time_seconds(soc_now: float, soc_target: float, charger_kw: float) -> float:
    soc_now = clamp(soc_now, 0.0, 1.0)
    soc_target = clamp(soc_target, 0.0, 1.0)
    if soc_target <= soc_now:
        return 0.0
    add_kwh = soc_to_kwh(soc_target - soc_now)
    hours = add_kwh / max(1e-6, charger_kw)
    return hours * 3600.0

def risk_penalty(p_success: float) -> float:
    eps = 1e-6
    p = clamp(p_success, eps, 1.0)
    return LAMBDA_RISK * (-math.log(p))


# =========================
# Data model
# =========================
@dataclass(frozen=True)
class Station:
    station_id: str
    lat: float
    lon: float

    @property
    def coord(self) -> Tuple[float, float]:
        return (self.lat, self.lon)

@dataclass
class LegResult:
    origin: Tuple[float, float]
    dest: Tuple[float, float]
    duration_s: float
    distance_m: float
    polyline: Optional[str] = None

@dataclass
class EvalResult:
    feasible: bool
    total_cost: float
    total_drive_s: float
    total_charge_s: float
    total_risk_penalty: float
    route: List[Tuple[str, Tuple[float, float]]]
    legs: List[LegResult]
    soc_timeline: List[float]
    arrival_times: List[datetime]


# =========================
# Load unique stations from backend csv
# =========================
def load_unique_stations(csv_path: str) -> Dict[str, Station]:
    df = pd.read_csv(csv_path)
    g = df.groupby("station_id", as_index=False)[["lat", "lon"]].first()
    stations = {}
    for _, r in g.iterrows():
        sid = str(r["station_id"])
        stations[sid] = Station(station_id=sid, lat=float(r["lat"]), lon=float(r["lon"]))
    return stations


# =========================
# Backend client
# =========================
def recommend_stations(
    lat: float,
    lon: float,
    when: datetime,
    k: int,
    radius_km: float,
    lambda_decay: float
) -> List[Tuple[str, float, float, float]]:
    params = {
        "lat": lat,
        "lon": lon,
        "time": fmt_backend_time(when),
        "k": k,
        "radius_km": radius_km,
        "lambda_decay": lambda_decay,
    }
    r = requests.get(BACKEND_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    # Try to find list payload robustly
    items = None
    if isinstance(data, dict):
        for key in ["recommendations", "topk", "results", "items"]:
            if key in data and isinstance(data[key], list):
                items = data[key]
                break
        if items is None:
            for v in data.values():
                if isinstance(v, list):
                    items = v
                    break
    elif isinstance(data, list):
        items = data

    if not items:
        return []

    out = []
    for it in items:
        sid = str(it.get("station_id") or it.get("id") or it.get("station") or "")
        slat = float(it.get("lat", it.get("latitude", np.nan)))
        slon = float(it.get("lon", it.get("longitude", np.nan)))
        p = float(it.get("p_success", it.get("probability", it.get("p", np.nan))))
        if sid and np.isfinite(slat) and np.isfinite(slon) and np.isfinite(p):
            out.append((sid, slat, slon, p))
    return out


# =========================
# Google Directions (traffic-aware)
# =========================
def google_directions_leg(origin: Tuple[float, float], dest: Tuple[float, float], departure_time: datetime) -> LegResult:
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": to_google_latlng(origin),
        "destination": to_google_latlng(dest),
        "mode": "driving",
        "departure_time": dt_to_unix_seconds(departure_time.replace(tzinfo=timezone.utc)),
        "alternatives": "false",
        "key": GOOGLE_API_KEY,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    if js.get("status") != "OK":
        raise RuntimeError(f"Google Directions error: {js.get('status')} - {js.get('error_message')}")

    route0 = js["routes"][0]
    leg0 = route0["legs"][0]
    dist_m = float(leg0["distance"]["value"])
    dur_s = float(leg0.get("duration_in_traffic", leg0["duration"])["value"])
    pl = route0.get("overview_polyline", {}).get("points")
    return LegResult(origin=origin, dest=dest, duration_s=dur_s, distance_m=dist_m, polyline=pl)


# =========================
# Evaluate solution (SOC + times + risk)
# =========================
def evaluate_solution(stop_ids: List[str], stations: Dict[str, Station], departure_time: datetime) -> EvalResult:
    route_nodes: List[Tuple[str, Tuple[float, float]]] = [("O", ORIGIN)]
    for sid in stop_ids:
        if sid not in stations:
            return EvalResult(False, 1e18, 0, 0, 0, [], [], [], [])
        route_nodes.append((sid, stations[sid].coord))
    route_nodes.append(("D", DESTINATION))

    soc = START_SOC
    t = departure_time

    legs: List[LegResult] = []
    soc_timeline: List[float] = [soc]
    arrival_times: List[datetime] = [t]

    total_drive = 0.0
    total_charge = 0.0
    total_risk = 0.0

    for i in range(len(route_nodes) - 1):
        _, a = route_nodes[i]
        nid_b, b = route_nodes[i + 1]

        leg = google_directions_leg(a, b, t)
        legs.append(leg)
        total_drive += leg.duration_s

        e_kwh = leg_energy_kwh(leg.distance_m)
        soc_after = soc - kwh_to_soc(e_kwh)
        if soc_after < MIN_SOC - 1e-9:
            return EvalResult(False, 1e18, total_drive, total_charge, total_risk,
                             route_nodes, legs, soc_timeline, arrival_times)

        # advance
        t = t + timedelta(seconds=leg.duration_s)
        soc = soc_after

        # if station: add risk + charge to TARGET_SOC_AT_CHARGE
        if nid_b != "D":
            try:
                recs = recommend_stations(b[0], b[1], t, k=K_RECOMMEND,
                                         radius_km=RADIUS_KM, lambda_decay=LAMBDA_DECAY)
            except Exception:
                recs = []

            p = None
            for sid, _, _, ps in recs:
                if sid == nid_b:
                    p = ps
                    break
            if p is None:
                p = 0.25

            total_risk += risk_penalty(p)

            soc_target = clamp(max(soc, TARGET_SOC_AT_CHARGE), 0.0, 1.0)
            ct = charge_time_seconds(soc, soc_target, CHARGER_KW_ASSUMED)
            total_charge += ct
            t = t + timedelta(seconds=ct)
            soc = soc_target

        soc_timeline.append(soc)
        arrival_times.append(t)

    total_cost = total_drive + total_charge + total_risk
    return EvalResult(True, total_cost, total_drive, total_charge, total_risk,
                      route_nodes, legs, soc_timeline, arrival_times)


# =========================
# Initial solution (greedy)
# =========================
def build_initial_solution(stations: Dict[str, Station], departure_time: datetime) -> List[str]:
    # Try direct first
    if evaluate_solution([], stations, departure_time).feasible:
        return []

    # Else: build feasible chain via reachability shortest path
    stops = build_feasible_chain_shortest_path(stations, departure_time)
    return stops


# =========================
# ALNS operators
# =========================
def destroy(stops: List[str]) -> List[str]:
    if not stops:
        return stops
    r = random.randint(DESTROY_MIN, min(DESTROY_MAX, len(stops)))
    idxs = sorted(random.sample(range(len(stops)), r), reverse=True)
    out = stops[:]
    for i in idxs:
        out.pop(i)
    return out

def repair(stops: List[str], stations: Dict[str, Station], departure_time: datetime) -> List[str]:
    res = evaluate_solution(stops, stations, departure_time)
    if res.feasible:
        return stops

    # midpoint between last stop (or origin) and destination
    a = ORIGIN if not stops else stations[stops[-1]].coord
    b = DESTINATION
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    qtime = departure_time + timedelta(hours=2)

    recs = recommend_stations(mid[0], mid[1], qtime, k=K_RECOMMEND,
                             radius_km=RADIUS_KM, lambda_decay=LAMBDA_DECAY)

    candidates = []
    for sid, slat, slon, _p in recs:
        if sid not in stations:
            stations[sid] = Station(sid, slat, slon)
        candidates.append(sid)

    if not candidates:
        return stops

    best = None
    best_cost = 1e18
    for sid in candidates[:8]:
        for pos in range(len(stops) + 1):
            trial = stops[:pos] + [sid] + stops[pos:]
            r = evaluate_solution(trial, stations, departure_time)
            if r.feasible and r.total_cost < best_cost:
                best_cost = r.total_cost
                best = trial

    return best if best is not None else stops


def alns_optimize(stations: Dict[str, Station], departure_time: datetime) -> EvalResult:
    current_stops = build_initial_solution(stations, departure_time)
    current = evaluate_solution(current_stops, stations, departure_time)

    best = current
    best_stops = current_stops[:]

    # simulated annealing
    T = 1e4
    cooling = 0.995

    for _ in range(ALNS_ITERS):
        base = best_stops if random.random() < 0.3 else current_stops
        partial = destroy(base)
        candidate_stops = repair(partial, stations, departure_time)
        cand = evaluate_solution(candidate_stops, stations, departure_time)

        if cand.total_cost < current.total_cost:
            accept = True
        else:
            delta = cand.total_cost - current.total_cost
            accept = (random.random() < math.exp(-delta / max(1e-9, T)))

        if accept:
            current_stops = candidate_stops
            current = cand

        if cand.feasible and cand.total_cost < best.total_cost:
            best = cand
            best_stops = candidate_stops[:]

        T *= cooling

    return best

def build_feasible_chain_shortest_path(stations: Dict[str, Station], departure_time: datetime) -> List[str]:
    """
    Build a feasible O->...->D chain using a time-dependent reachability graph.
    Since we only have 10 stations, we can brute pairwise legs with Google.
    """
    # Nodes: O, station ids, D
    node_ids = ["O"] + list(stations.keys()) + ["D"]

    def coord(nid: str) -> Tuple[float, float]:
        if nid == "O":
            return ORIGIN
        if nid == "D":
            return DESTINATION
        return stations[nid].coord

    # Precompute pairwise leg feasibility and base cost (drive time)
    # NOTE: This ignores that departure_time changes after charging; it’s a strong baseline for feasibility.
    edges = {nid: [] for nid in node_ids}

    for i in node_ids:
        for j in node_ids:
            if i == j:
                continue
            # don't go out of destination
            if i == "D":
                continue
            # don't go into origin
            if j == "O":
                continue

            a = coord(i)
            b = coord(j)

            # Get traffic-aware leg for baseline departure_time
            try:
                leg = google_directions_leg(a, b, departure_time)
            except Exception:
                continue

            e_kwh = leg_energy_kwh(leg.distance_m)
            soc_need = kwh_to_soc(e_kwh)

            # Feasible if you started this leg at TARGET_SOC_AT_CHARGE (or START_SOC if origin)
            soc_available = START_SOC if i == "O" else TARGET_SOC_AT_CHARGE
            if (soc_available - soc_need) < MIN_SOC:
                continue  # not reachable with buffer

            # Risk penalty only when arriving at a station (not destination)
            extra = 0.0
            if j not in ["O", "D"]:
                # use backend probability at baseline arrival time
                arr_time = departure_time + timedelta(seconds=leg.duration_s)
                p = 0.25
                try:
                    recs = recommend_stations(b[0], b[1], arr_time, k=K_RECOMMEND,
                                             radius_km=RADIUS_KM, lambda_decay=LAMBDA_DECAY)
                    for sid, _, _, ps in recs:
                        if sid == j:
                            p = ps
                            break
                except Exception:
                    pass
                extra = risk_penalty(p)

            edges[i].append((j, leg.duration_s + extra))

    # Dijkstra from O to D
    INF = 1e18
    dist = {nid: INF for nid in node_ids}
    parent = {nid: None for nid in node_ids}
    dist["O"] = 0.0
    pq = [(0.0, "O")]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u == "D":
            break
        for v, w in edges[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    if dist["D"] >= INF/2:
        return []  # no feasible chain

    # Reconstruct path
    path = []
    cur = "D"
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()

    # Convert to stop list (exclude O and D)
    stops = [nid for nid in path if nid not in ["O", "D"]]
    return stops


def main():
    print("Backend:", BACKEND_URL)
    print("CSV:", EV_CHARGING_CSV)

    stations = load_unique_stations(EV_CHARGING_CSV)
    print(f"Loaded unique stations: {len(stations)}")

    if DEPARTURE_TIME < datetime.now().replace(microsecond=0):
        raise ValueError("DEPARTURE_TIME must be now or future for traffic-aware routing.")

    best = alns_optimize(stations, DEPARTURE_TIME)

    if not best.feasible:
        print("No feasible route found. Try increasing RADIUS_KM or battery/SOC params.")
        return

    print("\n=== BEST ROUTE (ALNS) ===")
    print(f"Total cost: {best.total_cost/3600:.2f} h "
          f"(drive {best.total_drive_s/3600:.2f} h + "
          f"charge {best.total_charge_s/3600:.2f} h + "
          f"risk {best.total_risk_penalty:.2f})")

    print("\nStops (node, lat, lon, SOC, time):")
    for (nid, coord), soc, t in zip(best.route, best.soc_timeline, best.arrival_times):
        print(f"  {nid:12s} {coord[0]:.5f},{coord[1]:.5f}   SOC={soc*100:5.1f}%   time={t}")

    print("\nLegs:")
    for i, leg in enumerate(best.legs):
        km = leg.distance_m/1000
        mins = leg.duration_s/60
        print(f"  {i+1:02d}. {km:7.1f} km   {mins:7.1f} min")

    print("\nWaypoints (lat,lon):")
    for nid, coord in best.route:
        print(f"  {nid}: {coord[0]},{coord[1]}")


if __name__ == "__main__":
    main()
