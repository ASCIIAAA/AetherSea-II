from __future__ import annotations

import os
import sys
import logging
import datetime
from pathlib import Path

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AetherSea-II  |  Marine Debris Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── GEE availability check ───────────────────────────────────────────────────
_GEE_AVAILABLE = False
try:
    import ee
    from backend.fetch_satellite import (
        init_gee,
        get_plastic_tile_url,
        get_hotspots,
        get_region_stats,
        DEFAULT_AOI,
    )
    _GEE_AVAILABLE = True
except ImportError:
    pass

from backend.routing_engine import plan_cleanup_route

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Mono', monospace;
        background-color: #050d1a;
        color: #c8e6ff;
    }
    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        color: #00e5ff;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 0.85rem;
        color: #4fc3f7;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #0a1f3a 0%, #0d2b4e 100%);
        border: 1px solid #1565c0;
        border-radius: 8px;
        padding: 1rem 1.4rem;
        margin-bottom: 0.6rem;
    }
    .metric-label { font-size: 0.7rem; color: #78909c; letter-spacing: 2px; text-transform: uppercase; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #00e5ff; }
    .status-live   { color: #00e676; font-size: 0.75rem; }
    .status-cached { color: #ffd740; font-size: 0.75rem; }
    .status-demo   { color: #ff6e40; font-size: 0.75rem; }
    div[data-testid="stSidebar"] { background-color: #040c18; border-right: 1px solid #1565c0; }
    hr { border-color: #1565c0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar – controls
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️  Mission Control")
    st.markdown("---")

    date_range = st.date_input(
        "Analysis window",
        value=(datetime.date(2024, 1, 1), datetime.date(2024, 6, 30)),
    )
    start_str = str(date_range[0]) if len(date_range) > 0 else "2024-01-01"
    end_str   = str(date_range[1]) if len(date_range) > 1 else "2024-06-30"

    fdi_thresh = st.slider("FDI detection threshold", 0.01, 0.15, 0.04, 0.005,
                            help="Floating Debris Index minimum – raise to reduce false positives")

    ship_speed = st.slider("Vessel speed (knots)", 5, 25, 12,
                            help="Calm-water cruising speed of cleanup vessel")

    max_hotspots = st.slider("Max hotspots to load", 10, 200, 80,
                              help="Server-side cap on GEE sample size")

    show_tiles   = st.toggle("Show FDI satellite overlay", value=True)
    show_currents = st.toggle("Show NOAA current vectors", value=True)
    show_route   = st.toggle("Show optimised cleanup route", value=True)

    run_btn = st.button("🚀  Run Analysis", use_container_width=True, type="primary")

    st.markdown("---")
    if _GEE_AVAILABLE:
        st.markdown('<span class="status-live">● GEE connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-demo">● Demo mode (no GEE auth)</span>', unsafe_allow_html=True)
        st.caption("Set EE_SERVICE_ACCOUNT + EE_KEY_FILE env vars to enable live data.")


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════

c1, c2 = st.columns([3, 1])
with c1:
    st.markdown('<p class="main-title">🌊 AetherSea-II</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Marine Debris Intelligence Platform &nbsp;·&nbsp; Arabian Sea</p>',
                unsafe_allow_html=True)
with c2:
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d  %H:%M UTC")
    st.markdown(f"<div style='text-align:right;color:#4fc3f7;font-size:0.75rem;padding-top:1rem'>{ts}</div>",
                unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# Data pipeline
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _load_live_hotspots(start, end, thresh, max_pts):
    """Pull real hotspots from GEE (cached 1 hour)."""
    init_gee()
    return get_hotspots(start=start, end=end,
                        fdi_thresh=thresh, max_points=max_pts)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_tile_url(start, end):
    init_gee()
    return get_plastic_tile_url(start=start, end=end)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_region_stats(start, end):
    init_gee()
    return get_region_stats(start=start, end=end)


def _demo_hotspots(n: int = 60) -> list[dict]:
    rng = np.random.default_rng(seed=42)
    # Bias coordinates toward known high-debris zones
    centres = [
        (15.0, 65.0), (22.0, 63.0), (12.0, 72.0),
        (10.0, 68.0), (18.0, 70.0), (25.0, 66.0),
    ]
    hotspots = []
    per_cluster = n // len(centres)
    for (clat, clon) in centres:
        for _ in range(per_cluster):
            lat = float(rng.normal(clat, 1.5))
            lon = float(rng.normal(clon, 1.8))
            lat = max(5.0, min(30.0, lat))
            lon = max(60.0, min(80.0, lon))
            fdi = float(rng.uniform(0.04, 0.14))
            hotspots.append({"lat": round(lat,4), "lon": round(lon,4),
                             "fdi": round(fdi,5), "pi": round(fdi*0.8, 5)})
    return hotspots


# ── Trigger on button OR first load ──────────────────────────────────────────
if "hotspots" not in st.session_state or run_btn:
    with st.spinner("🛰️  Querying satellite pipeline…"):
        if _GEE_AVAILABLE:
            try:
                st.session_state["hotspots"]    = _load_live_hotspots(
                    start_str, end_str, fdi_thresh, max_hotspots)
                st.session_state["region_stats"] = _load_region_stats(start_str, end_str)
                st.session_state["tile_info"]    = _load_tile_url(start_str, end_str) if show_tiles else None
                st.session_state["data_source"]  = "live"
            except Exception as exc:
                logger.warning("GEE call failed (%s). Falling back to demo data.", exc)
                st.warning(f"⚠️ GEE returned an error – showing demo data.\n`{exc}`")
                st.session_state["hotspots"]    = _demo_hotspots(max_hotspots)
                st.session_state["region_stats"] = {"mean_fdi": 0.065, "mean_pi": 0.052}
                st.session_state["tile_info"]    = None
                st.session_state["data_source"]  = "demo"
        else:
            st.session_state["hotspots"]    = _demo_hotspots(max_hotspots)
            st.session_state["region_stats"] = {"mean_fdi": 0.065, "mean_pi": 0.052}
            st.session_state["tile_info"]    = None
            st.session_state["data_source"]  = "demo"

hotspots    = st.session_state["hotspots"]
region_stats = st.session_state.get("region_stats", {})
tile_info    = st.session_state.get("tile_info")
data_source  = st.session_state.get("data_source", "demo")

# ── Routing ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def _plan_route(hs_json, speed):
    import json
    hs = json.loads(hs_json)
    return plan_cleanup_route(hs, ship_speed=speed)

import json
route = _plan_route(json.dumps(hotspots), ship_speed)


# ══════════════════════════════════════════════════════════════════════════════
# KPI row
# ══════════════════════════════════════════════════════════════════════════════

k1, k2, k3, k4, k5 = st.columns(5)

src_badge = {
    "live":  '<span class="status-live">● LIVE GEE</span>',
    "demo":  '<span class="status-demo">● DEMO</span>',
    "cached":'<span class="status-cached">● CACHED</span>',
}.get(data_source, "")

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Hotspots detected {src_badge}</div>
        <div class="metric-value">{len(hotspots)}</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Mean FDI index</div>
        <div class="metric-value">{region_stats.get('mean_fdi', 0):.4f}</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Route distance (km)</div>
        <div class="metric-value">{route['total_dist_km']:,.0f}</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Est. time at sea (h)</div>
        <div class="metric-value">{route['total_cost']:,.1f}</div>
    </div>""", unsafe_allow_html=True)

with k5:
    boosts = [s["current_boost"] for s in route["segments"]]
    avg_boost = sum(boosts)/len(boosts) if boosts else 0
    colour = "#00e676" if avg_boost >= 0 else "#ff5252"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg current assist (kts)</div>
        <div class="metric-value" style="color:{colour}">{avg_boost:+.2f}</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Map
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 🗺️  Live Debris Intelligence Map")

m = folium.Map(
    location=[17.5, 70.0],
    zoom_start=5,
    tiles="CartoDB dark_matter",
    attr="CartoDB",
)

# ── FDI satellite tile overlay ────────────────────────────────────────────────
if show_tiles and tile_info:
    folium.TileLayer(
        tiles       = tile_info["tile_url"],
        attr        = tile_info["attribution"],
        name        = tile_info["name"],
        opacity     = 0.6,
        overlay     = True,
        show        = True,
    ).add_to(m)

# ── NOAA current vectors (sampled grid) ───────────────────────────────────────
if show_currents:
    try:
        from backend.routing_engine import _get_current_field
        cf = _get_current_field()
        # Draw arrows on a coarse 2° grid
        for lat in np.arange(7, 29, 2.5):
            for lon in np.arange(62, 79, 2.5):
                u, v = cf.get_uv(lat, lon)
                speed_ms = math.hypot(u, v)
                if speed_ms < 0.05:
                    continue
                # Arrow endpoint (scaled for visibility)
                scale = 1.2
                dlat = v * scale
                dlon = u * scale / max(math.cos(math.radians(lat)), 0.1)
                arrow_colour = "#29b6f6" if speed_ms < 0.3 else "#e91e63"
                folium.PolyLine(
                    [[lat, lon], [lat+dlat, lon+dlon]],
                    color=arrow_colour, weight=1.5, opacity=0.7,
                    tooltip=f"u={u:.2f} m/s  v={v:.2f} m/s",
                ).add_to(m)
    except Exception as exc:
        logger.debug("Current overlay skipped: %s", exc)

import math  # ensure available in scope (already imported above)

# ── Hotspot markers ───────────────────────────────────────────────────────────
waypoint_idx = {id(h): i for i, h in enumerate(route["waypoints"])}

for h in hotspots:
    intensity = min(h["fdi"] / 0.15, 1.0)
    r = int(255 * intensity)
    g = int(100 * (1 - intensity))
    b = 50
    colour = f"#{r:02x}{g:02x}{b:02x}"
    rank   = waypoint_idx.get(id(h))
    popup  = (f"<b>FDI:</b> {h['fdi']:.5f}<br>"
              f"<b>PI:</b>  {h.get('pi',0):.5f}<br>"
              f"<b>Lat/Lon:</b> {h['lat']:.3f}, {h['lon']:.3f}")
    if rank is not None:
        popup += f"<br><b>Route stop:</b> #{rank+1}"

    folium.CircleMarker(
        location=[h["lat"], h["lon"]],
        radius=5 + intensity * 6,
        color=colour, fill=True, fill_color=colour, fill_opacity=0.75,
        tooltip=f"FDI {h['fdi']:.4f}",
        popup=folium.Popup(popup, max_width=220),
    ).add_to(m)

# ── Cleanup route polyline ────────────────────────────────────────────────────
if show_route and route["waypoints"]:
    coords = [[w["lat"], w["lon"]] for w in route["waypoints"]]

    folium.PolyLine(
        coords,
        color="#00e5ff", weight=2.5, opacity=0.85,
        tooltip="Optimised cleanup route",
        dash_array="6 4",
    ).add_to(m)

    # Number waypoints
    for i, w in enumerate(route["waypoints"]):
        folium.Marker(
            location=[w["lat"], w["lon"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:9px;background:#00e5ff;color:#000;'
                     f'border-radius:50%;width:18px;height:18px;'
                     f'display:flex;align-items:center;justify-content:center;'
                     f'font-weight:700">{i+1}</div>',
                icon_size=(18, 18), icon_anchor=(9, 9),
            ),
        ).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, width="100%", height=560, returned_objects=[])


# ══════════════════════════════════════════════════════════════════════════════
# Route table + segment stats
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
col_a, col_b = st.columns([2, 1])

with col_a:
    st.markdown("### 🗒️  Route Manifest")
    if route["waypoints"]:
        df = pd.DataFrame([
            {
                "Stop":       i+1,
                "Lat":        w["lat"],
                "Lon":        w["lon"],
                "FDI":        w["fdi"],
                "Dist to next (km)": (route["segments"][i]["dist_km"]
                                      if i < len(route["segments"]) else "–"),
                "Current (kts)":     (route["segments"][i]["current_boost"]
                                      if i < len(route["segments"]) else "–"),
                "Leg time (h)":      (route["segments"][i]["cost"]
                                      if i < len(route["segments"]) else "–"),
            }
            for i, w in enumerate(route["waypoints"])
        ])
        st.dataframe(df, use_container_width=True, height=340)

with col_b:
    st.markdown("### 📊  Current Efficiency")
    if route["segments"]:
        segs = route["segments"]
        tailwind = sum(1 for s in segs if s["current_boost"] > 0.1)
        headwind = sum(1 for s in segs if s["current_boost"] < -0.1)
        neutral  = len(segs) - tailwind - headwind

        st.metric("Tailwind legs",  tailwind,  delta="⬇️ fuel saved",   delta_color="normal")
        st.metric("Headwind legs",  headwind,  delta="⬆️ fuel penalty",  delta_color="inverse")
        st.metric("Neutral legs",   neutral)
        st.metric("Total route hours", f"{route['total_cost']:.1f} h")
        st.metric("Total distance",    f"{route['total_dist_km']:,.0f} km")


# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#37474f;font-size:0.7rem'>"
    "AetherSea-II &nbsp;·&nbsp; Sentinel-2 via Google Earth Engine &nbsp;·&nbsp; "
    "NOAA OSCAR Currents &nbsp;·&nbsp; "
    "Plastic detection: FDI + PI indices &nbsp;·&nbsp; "
    "Routing: Hydraulic Friction-Aware 2-opt Dijkstra"
    "</div>",
    unsafe_allow_html=True,
)
