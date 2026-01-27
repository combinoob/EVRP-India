from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from src.charging.availability_client import recommend, stations_nearby, StationReco
from src.routing.google_api import google_fastest_route, energy_for_google_route_kwh
from src.routing.google_energy_profile import compute_step_energies_from_google_steps, build_soc_track

LatLng = Tuple[float, float]


@dataclass
class PlannedStop:
    station_id: str
    lat: float
    lon: float
    p_success: float
    expected_wait_min: float
    charge_min: float


@dataclass
class TripPlan:
    total_time_min: float
    total_energy_kwh: float
    route_legs: List[List[LatLng]]
    chosen_stops: List[PlannedStop]
    nearby_stations: List[Dict]  # stations along corridor; may include p_success
    soc_track: List[Dict]


def expected_wait_min(p_success: float, wait_penalty_min: float) -> float:
    p = max(0.0, min(1.0, float(p_success)))
    return float(wait_penalty_min) * (1.0 - p)


def charge_time_min(energy_kwh: float, charger_kw: float) -> float:
    return (max(0.0, float(energy_kwh)) / max(1e-6, float(charger_kw))) * 60.0


def attach_psuccess_to_nearby_stations(stations: List[Dict], recos: List[StationReco]) -> List[Dict]:
    mp = {r.station_id: float(r.p_success) for r in recos}
    out: List[Dict] = []
    for st in stations:
        st2 = dict(st)
        sid = st2.get("station_id")
        if sid in mp:
            st2["p_success"] = mp[sid]
        out.append(st2)
    return out


def collect_stations_along_polyline(
    availability_base_url: str,
    polyline: List[LatLng],
    radius_km: float,
    sample_every_n_points: int = 25,
) -> List[Dict]:
    """
    Collect charging stations in a corridor around the route by sampling points along the polyline.
    Dedupe by station_id.
    """
    if not polyline:
        return []

    merged: Dict[str, Dict] = {}

    n = max(1, int(sample_every_n_points))
    idxs = list(range(0, len(polyline), n))
    if idxs[-1] != len(polyline) - 1:
        idxs.append(len(polyline) - 1)

    for i in idxs:
        lat, lon = polyline[i]
        try:
            near = stations_nearby(availability_base_url, lat, lon, radius_km=radius_km)
        except Exception:
            continue

        for st in near:
            sid = st.get("station_id")
            if sid:
                merged[sid] = st

    return list(merged.values())


def merge_station_lists(*lists: List[Dict]) -> List[Dict]:
    merged: Dict[str, Dict] = {}
    for lst in lists:
        for st in lst:
            sid = st.get("station_id")
            if sid:
                merged[sid] = st
    return list(merged.values())


def plan_min_time_with_one_stop(
    *,
    google_api_key: str,
    availability_base_url: str,
    origin: LatLng,
    destination: LatLng,
    departure_dt: datetime,
    vehicle_mass_kg: float,
    battery_kwh: float,
    soc_start: float,
    soc_min: float,
    charge_target_soc: float,
    charger_power_kw_default: float,
    wait_penalty_min_: float,
    p_success_min: float,
    lookback_km: float,
    sample_every_km: float,
    radius_km: float = 10.0,
    reco_k: int = 3,
    lambda_decay: float = 0.05,
    energy_cfg: dict | None = None,
) -> TripPlan:
    if energy_cfg is None:
        raise ValueError("energy_cfg required")

    usable_kwh = battery_kwh * float(soc_start)
    reserve_kwh = battery_kwh * float(soc_min)

    # -------------------------
    # Direct fastest route
    # -------------------------
    direct = google_fastest_route(google_api_key, origin, destination, departure_time="now")
    direct_energy = float(
        energy_for_google_route_kwh(
            api_key=google_api_key,
            steps=direct["steps"],
            vehicle_mass_kg=vehicle_mass_kg,
            energy_cfg=energy_cfg,
            cache_dir="data/cache",
        )
    )
    direct_time_min = float(direct["duration_s"]) / 60.0

    # Step energies for SOC track and crossing detection
    step_energies = compute_step_energies_from_google_steps(
        api_key=google_api_key,
        steps=direct["steps"],
        vehicle_mass_kg=vehicle_mass_kg,
        energy_cfg=energy_cfg,
        cache_dir="data/cache",
    )
    soc_track_direct = build_soc_track(step_energies, battery_kwh=battery_kwh, soc_start=soc_start)

    # -------------------------
    # If direct is feasible
    # -------------------------
    if direct_energy <= max(0.0, usable_kwh - reserve_kwh):
        # Collect stations along the WHOLE route corridor
        corridor_stations = collect_stations_along_polyline(
            availability_base_url=availability_base_url,
            polyline=direct["path_latlng"],
            radius_km=radius_km,
            sample_every_n_points=25,
        )

        # Optional: attach p_success to stations by querying recommend at origin/time
        # (This colors stations near origin; corridor points may differ slightly, but still helpful)
        try:
            recos0 = recommend(
                availability_base_url,
                lat=origin[0],
                lon=origin[1],
                when=departure_dt,
                k=200,  # large so many stations get p_success
                radius_km=radius_km,
                lambda_decay=lambda_decay,
            )
            corridor_stations = attach_psuccess_to_nearby_stations(corridor_stations, recos0)
        except Exception:
            pass

        return TripPlan(
            total_time_min=direct_time_min,
            total_energy_kwh=direct_energy,
            route_legs=[direct["path_latlng"]],
            chosen_stops=[],
            nearby_stations=corridor_stations,
            soc_track=soc_track_direct,
        )

    # -------------------------
    # Direct NOT feasible -> find SOC crossing step index
    # -------------------------
    remaining = usable_kwh
    cross_i = None
    for i, se in enumerate(step_energies):
        remaining -= float(se.e_kwh)
        if remaining <= reserve_kwh:
            cross_i = i
            break
    if cross_i is None:
        cross_i = max(0, len(step_energies) - 1)

    # Candidate points: go backward by distance
    cum_m: List[float] = []
    dist_sum = 0.0
    for se in step_energies:
        dist_sum += float(se.dist_m)
        cum_m.append(dist_sum)

    cross_dist_m = cum_m[cross_i] if cum_m else 0.0
    lookback_m = float(lookback_km) * 1000.0
    sample_m = max(300.0, float(sample_every_km) * 1000.0)

    candidates: List[int] = []
    d = cross_dist_m
    while d >= max(0.0, cross_dist_m - lookback_m):
        j = next((idx for idx, cm in enumerate(cum_m) if cm >= d), None)
        if j is not None:
            candidates.append(j)
        d -= sample_m

    # dedupe
    seen = set()
    cand: List[int] = []
    for j in candidates:
        if j not in seen:
            seen.add(j)
            cand.append(j)

    best = None  # (total_time_min, total_energy, leg1_poly, leg2_poly, stop, stations_for_plot, soc_track)

    # -------------------------
    # Evaluate candidates
    # -------------------------
    for j in cand:
        q_lat = step_energies[j].step_end_lat
        q_lon = step_energies[j].step_end_lon
        q_dt = departure_dt + timedelta(seconds=sum(se.dt_s for se in step_energies[: j + 1]))

        try:
            recos = recommend(
                availability_base_url,
                lat=q_lat,
                lon=q_lon,
                when=q_dt,
                k=reco_k,
                radius_km=radius_km,
                lambda_decay=lambda_decay,
            )
        except Exception:
            continue

        recos = [r for r in recos if float(r.p_success) >= float(p_success_min)]
        if not recos:
            continue

        # station coords near query point
        st_near = stations_nearby(availability_base_url, q_lat, q_lon, radius_km=radius_km)
        coord = {st["station_id"]: (float(st["lat"]), float(st["lon"])) for st in st_near if "station_id" in st}

        # attach p_success so map can color these stations
        st_near = attach_psuccess_to_nearby_stations(st_near, recos)

        for r in recos:
            if r.station_id not in coord:
                continue
            st_lat, st_lon = coord[r.station_id]

            # route origin -> station
            leg1 = google_fastest_route(google_api_key, origin, (st_lat, st_lon), departure_time="now")
            e1 = float(
                energy_for_google_route_kwh(
                    api_key=google_api_key,
                    steps=leg1["steps"],
                    vehicle_mass_kg=vehicle_mass_kg,
                    energy_cfg=energy_cfg,
                    cache_dir="data/cache",
                )
            )
            if e1 > max(0.0, usable_kwh - reserve_kwh):
                continue

            # route station -> destination
            leg2 = google_fastest_route(google_api_key, (st_lat, st_lon), destination, departure_time="now")
            e2 = float(
                energy_for_google_route_kwh(
                    api_key=google_api_key,
                    steps=leg2["steps"],
                    vehicle_mass_kg=vehicle_mass_kg,
                    energy_cfg=energy_cfg,
                    cache_dir="data/cache",
                )
            )

            remaining_after_leg1 = usable_kwh - e1
            target_kwh = battery_kwh * float(charge_target_soc)
            needed_depart_station = e2 + reserve_kwh

            charge_to_kwh = max(needed_depart_station, min(target_kwh, battery_kwh))
            energy_to_charge = max(0.0, charge_to_kwh - remaining_after_leg1)

            chg_min = charge_time_min(energy_to_charge, charger_power_kw_default)
            wait_min = expected_wait_min(r.p_success, wait_penalty_min_)

            total_time = (leg1["duration_s"] + leg2["duration_s"]) / 60.0 + wait_min + chg_min
            total_energy = e1 + e2

            stop = PlannedStop(
                station_id=r.station_id,
                lat=st_lat,
                lon=st_lon,
                p_success=float(r.p_success),
                expected_wait_min=float(wait_min),
                charge_min=float(chg_min),
            )

            # SOC track: leg1 steps -> charge jump -> leg2 steps
            leg1_steps = compute_step_energies_from_google_steps(
                api_key=google_api_key,
                steps=leg1["steps"],
                vehicle_mass_kg=vehicle_mass_kg,
                energy_cfg=energy_cfg,
                cache_dir="data/cache",
            )
            soc_track = build_soc_track(leg1_steps, battery_kwh=battery_kwh, soc_start=soc_start)

            t_last = soc_track[-1]["t_s"] if soc_track else 0.0
            soc_track.append(
                {"lat": st_lat, "lon": st_lon, "soc": float(charge_target_soc), "t_s": t_last + float(chg_min) * 60.0}
            )

            leg2_steps = compute_step_energies_from_google_steps(
                api_key=google_api_key,
                steps=leg2["steps"],
                vehicle_mass_kg=vehicle_mass_kg,
                energy_cfg=energy_cfg,
                cache_dir="data/cache",
            )
            soc2 = build_soc_track(leg2_steps, battery_kwh=battery_kwh, soc_start=charge_target_soc)
            if soc2:
                base_t = soc_track[-1]["t_s"]
                for p in soc2:
                    soc_track.append({"lat": p["lat"], "lon": p["lon"], "soc": p["soc"], "t_s": base_t + p["t_s"]})

            if best is None or total_time < best[0]:
                best = (total_time, total_energy, leg1["path_latlng"], leg2["path_latlng"], stop, st_near, soc_track)

    if best is None:
        raise RuntimeError("No feasible charging stop found. Increase radius/lookback or lower p_success threshold.")

    total_time_min, total_energy, leg1_poly, leg2_poly, stop, st_near_for_plot, soc_track = best

    # -------------------------
    # NEW: collect stations along the ENTIRE FINAL ROUTE CORRIDOR
    # -------------------------
    corridor_leg1 = collect_stations_along_polyline(
        availability_base_url=availability_base_url,
        polyline=leg1_poly,
        radius_km=radius_km,
        sample_every_n_points=25,
    )
    corridor_leg2 = collect_stations_along_polyline(
        availability_base_url=availability_base_url,
        polyline=leg2_poly,
        radius_km=radius_km,
        sample_every_n_points=25,
    )

    # attach p_success from the candidate-point recommend response (where available)
    corridor_leg1 = attach_psuccess_to_nearby_stations(corridor_leg1, recommend(
        availability_base_url, lat=origin[0], lon=origin[1], when=departure_dt, k=200, radius_km=radius_km, lambda_decay=lambda_decay
    )) if corridor_leg1 else corridor_leg1

    # merge everything for map: corridor stations + stations near chosen point (with p_success)
    all_for_plot = merge_station_lists(corridor_leg1, corridor_leg2, st_near_for_plot)

    return TripPlan(
        total_time_min=float(total_time_min),
        total_energy_kwh=float(total_energy),
        route_legs=[leg1_poly, leg2_poly],
        chosen_stops=[stop],
        nearby_stations=all_for_plot,
        soc_track=soc_track,
    )