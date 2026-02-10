import math
from typing import List, Tuple

import folium
import polyline as polyline_lib
import pandas as pd

# Import from your existing solver file
# Run as: python -m experiments.visualize_blr_chn_map
import experiments.blr_to_chn_alns as solver


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(x))


def decode_leg_points(leg_polyline: str) -> List[Tuple[float, float]]:
    if not leg_polyline:
        return []
    pts = polyline_lib.decode(leg_polyline)
    return [(float(lat), float(lon)) for lat, lon in pts]


def stitch_route_points(legs: List[solver.LegResult]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for lg in legs:
        pts = decode_leg_points(lg.polyline)
        if not pts:
            pts = [lg.origin, lg.dest]
        if points and pts:
            if points[-1] == pts[0]:
                points.extend(pts[1:])
            else:
                points.extend(pts)
        else:
            points.extend(pts)
    return points


def cumulative_distances(points: List[Tuple[float, float]]) -> List[float]:
    if not points:
        return []
    cum = [0.0]
    for i in range(1, len(points)):
        cum.append(cum[-1] + haversine_m(points[i - 1], points[i]))
    return cum


def sample_points_by_distance(points: List[Tuple[float, float]], every_m: float = 3000.0):
    """Return sampled indices roughly every N meters along the polyline."""
    if not points:
        return [], []
    cum = cumulative_distances(points)
    samples = [0]
    next_d = every_m
    i = 1
    while i < len(points):
        if cum[i] >= next_d:
            samples.append(i)
            next_d += every_m
        i += 1
    if samples[-1] != len(points) - 1:
        samples.append(len(points) - 1)
    return samples, cum


def approx_soc_along_polyline(
    legs: List[solver.LegResult],
    soc_at_nodes: List[float],
):
    """
    Compute SOC along the stitched route by interpolating SOC on each leg polyline
    between node SOC values. Charging will appear as a 'jump' at stop markers, which is
    correct for presentation.
    """
    stitched_soc = []
    stitched_pts = []

    for i, lg in enumerate(legs):
        pts = decode_leg_points(lg.polyline)
        if not pts:
            pts = [lg.origin, lg.dest]

        cum = cumulative_distances(pts)
        total = cum[-1] if cum else 1.0

        soc_start = soc_at_nodes[i]
        soc_end = soc_at_nodes[i + 1]

        for k, p in enumerate(pts):
            frac = (cum[k] / total) if total > 0 else 0.0
            soc = soc_start + (soc_end - soc_start) * frac
            stitched_pts.append(p)
            stitched_soc.append(soc)

    return stitched_pts, stitched_soc


def load_all_stations_unique(csv_path: str):
    df = pd.read_csv(csv_path)
    g = df.groupby("station_id", as_index=False)[["lat", "lon"]].first()
    return [(str(r["station_id"]), float(r["lat"]), float(r["lon"])) for _, r in g.iterrows()]


def make_map(best: solver.EvalResult, out_html: str = "routes_soc_map.html"):
    # Route polyline points
    route_points = stitch_route_points(best.legs)
    if not route_points:
        route_points = [solver.ORIGIN, solver.DESTINATION]

    # Compute SOC along route polyline (approx)
    stitched_pts, stitched_soc = approx_soc_along_polyline(best.legs, best.soc_timeline)

    # Center map
    avg_lat = sum(p[0] for p in route_points) / len(route_points)
    avg_lon = sum(p[1] for p in route_points) / len(route_points)

    m = folium.Map(location=(avg_lat, avg_lon), zoom_start=8, tiles="cartodbpositron")

    # Determine used station IDs
    used_station_ids = {nid for nid, _ in best.route if nid not in ["O", "D"]}

    # Origin marker
    folium.Marker(
        location=solver.ORIGIN,
        icon=folium.Icon(color="blue", icon="play"),
        tooltip="Origin (Bengaluru)",
    ).add_to(m)

    # Destination marker
    folium.Marker(
        location=solver.DESTINATION,
        icon=folium.Icon(color="green", icon="flag"),
        tooltip="Destination (Chennai)",
    ).add_to(m)

    # All charging stations as marker icons:
    # Used stations -> red bolt
    # Unused stations -> gray bolt
    all_st = load_all_stations_unique(solver.EV_CHARGING_CSV)
    for sid, lat, lon in all_st:
        if sid in used_station_ids:
            color = "red"
            tip = f"Charging Stop (USED): {sid}"
        else:
            color = "gray"
            tip = f"Charging Station: {sid}"

        folium.Marker(
            location=(lat, lon),
            icon=folium.Icon(color=color, icon="bolt", prefix="fa"),
            tooltip=tip,
        ).add_to(m)

    # Draw route line
    folium.PolyLine(
        route_points,
        weight=5,
        opacity=0.8,
        tooltip="Traffic-aware route (Google Directions)",
    ).add_to(m)

    # SOC hover points (sampled to avoid clutter)
    if stitched_pts and stitched_soc:
        idxs, _cum = sample_points_by_distance(stitched_pts, every_m=4000.0)  # 4 km
        for idx in idxs:
            p = stitched_pts[idx]
            soc = stitched_soc[idx]
            folium.CircleMarker(
                location=p,
                radius=3,
                weight=1,
                fill=True,
                fill_opacity=0.8,
                tooltip=f"SOC ≈ {soc * 100:.1f}%",
            ).add_to(m)

    # Fit bounds
    m.fit_bounds(route_points)

    m.save(out_html)
    print(f"Saved map -> {out_html}")


def main():
    # Run solver to get best route (same as your CLI output)
    stations = solver.load_unique_stations(solver.EV_CHARGING_CSV)
    best = solver.alns_optimize(stations, solver.DEPARTURE_TIME)
    if not best.feasible:
        print("No feasible route. Can't visualize.")
        return
    make_map(best, out_html="routes_soc_map.html")


if __name__ == "__main__":
    main()