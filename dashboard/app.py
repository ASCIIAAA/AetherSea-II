from __future__ import annotations

import os
import sys
import math
import json
import logging
import datetime
from pathlib import Path

from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────
# ENV LOAD
# ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────
# LIBRARIES
# ─────────────────────────────────────────────────────────────
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
# AGENTS
# ─────────────────────────────────────────────────────────────
from agents.supervisor_agent import SupervisorAgent

# ─────────────────────────────────────────────────────────────
# BACKEND
# ─────────────────────────────────────────────────────────────
from backend.routing_engine import plan_cleanup_route

# ─────────────────────────────────────────────────────────────
# STREAMLIT CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AetherSea-II | Marine Debris Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# GEE CHECK
# ─────────────────────────────────────────────────────────────
_GEE_AVAILABLE = False
gee_error_msg = None

try:
    import ee

    from data.fetch_satellite import (
        init_gee,
        get_plastic_tile_url,
        get_region_stats,
        get_cloud_reduced_hotspots,
    )

    init_gee()

    _GEE_AVAILABLE = True
    logger.info("Earth Engine connected successfully.")

except Exception as e:
    _GEE_AVAILABLE = False
    gee_error_msg = f"{type(e).__name__}: {str(e)}"
    logger.error(gee_error_msg)

# ─────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #050d1a;
    color: white;
}

.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #00e5ff;
}

.metric-card {
    background: #0d1b2a;
    border: 1px solid #1565c0;
    border-radius: 10px;
    padding: 1rem;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    st.title("⚙️ Mission Control")

    date_range = st.date_input(
        "Analysis Window",
        value=(datetime.date(2024, 1, 1), datetime.date(2024, 6, 30)),
    )

    start_str = str(date_range[0])
    end_str = str(date_range[1])

    fdi_thresh = st.slider(
        "FDI Threshold",
        0.01,
        0.15,
        0.04,
        0.005
    )

    ship_speed = st.slider(
        "Vessel Speed",
        5,
        25,
        12
    )

    max_hotspots = st.slider(
        "Max Hotspots",
        10,
        200,
        80
    )

    show_tiles = st.toggle("Satellite Overlay", value=True)
    show_route = st.toggle("Cleanup Route", value=True)
    show_currents = st.toggle("Ocean Currents", value=True)

    run_btn = st.button(
        "🚀 Run Analysis",
        use_container_width=True,
        type="primary"
    )

    st.markdown("---")

    if _GEE_AVAILABLE:
        st.success("GEE Connected")
    else:
        st.error("Demo Mode")
        if gee_error_msg:
            st.caption(gee_error_msg)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<p class="main-title">🌊 AetherSea-II</p>',
    unsafe_allow_html=True
)

st.caption("Marine Debris Intelligence Platform")

# ─────────────────────────────────────────────────────────────
# DEMO HOTSPOTS
# ─────────────────────────────────────────────────────────────
def _demo_hotspots(
    n: int = 60,
    fdi_thresh: float = 0.04,
    start_date=None
):

    seed_val = hash(str(start_date)) % (10**8)

    rng = np.random.default_rng(seed=seed_val)

    centres = [
        (15.0, 65.0),
        (22.0, 63.0),
        (12.0, 72.0),
        (10.0, 68.0)
    ]

    hotspots = []

    per_cluster = max(1, n // len(centres))

    for clat, clon in centres:

        for _ in range(per_cluster):

            lat = float(rng.normal(clat, 1.5))
            lon = float(rng.normal(clon, 1.8))
            fdi = float(rng.uniform(fdi_thresh, 0.15))

            hotspots.append({
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "fdi": round(fdi, 5),
                "pi": round(fdi * 0.8, 5)
            })

    return hotspots[:n]

# ─────────────────────────────────────────────────────────────
# LOAD HOTSPOTS
# ─────────────────────────────────────────────────────────────
if "hotspots" not in st.session_state or run_btn:

    with st.spinner("Querying satellite systems..."):

        if _GEE_AVAILABLE:

            try:

                hotspots = get_cloud_reduced_hotspots(
                    lon_range=[60.0, 80.0],
                    lat_range=[5.0, 30.0],
                    start_date=start_str,
                    end_date=end_str,
                    fdi_threshold=fdi_thresh,
                    ndvi_threshold=0.15
                )

                st.session_state["hotspots"] = hotspots

                st.session_state["tile_info"] = get_plastic_tile_url(
                    aoi=ee.Geometry.Rectangle([60, 5, 80, 30]),
                    start=start_str,
                    end=end_str
                )

                st.session_state["data_source"] = "live"

            except Exception as e:

                logger.error(e)

                st.session_state["hotspots"] = _demo_hotspots(
                    max_hotspots,
                    fdi_thresh,
                    start_str
                )

                st.session_state["tile_info"] = None
                st.session_state["data_source"] = "demo"

        else:

            st.session_state["hotspots"] = _demo_hotspots(
                max_hotspots,
                fdi_thresh,
                start_str
            )

            st.session_state["tile_info"] = None
            st.session_state["data_source"] = "demo"

# ─────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────
hotspots = st.session_state["hotspots"]
tile_info = st.session_state["tile_info"]
data_source = st.session_state["data_source"]

# ─────────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def _plan_route(hs_json, speed):

    hs = json.loads(hs_json)

    if not hs:
        return {
            "waypoints": [],
            "segments": [],
            "total_cost": 0.0,
            "total_dist_km": 0.0
        }

    return plan_cleanup_route(
        hs,
        ship_speed=speed
    )

route = _plan_route(
    json.dumps(hotspots),
    ship_speed
)

# ─────────────────────────────────────────────────────────────
# AI SUPERVISOR AGENT
# ─────────────────────────────────────────────────────────────
agent = SupervisorAgent()

mission_summary = agent.generate_mission_report(
    hotspots=hotspots,
    route=route,
    source=data_source
)

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Hotspots", len(hotspots))

with k2:
    st.metric("Route Distance", f"{route['total_dist_km']:.0f} km")

with k3:
    st.metric("Mission Time", f"{route['total_cost']:.1f} hrs")

with k4:
    st.metric("Data Source", data_source.upper())

# ─────────────────────────────────────────────────────────────
# AI PANEL
# ─────────────────────────────────────────────────────────────
st.markdown("---")

st.subheader("🧠 Supervisor AI Assessment")

st.info(mission_summary)

# ─────────────────────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────────────────────
st.markdown("### 🗺️ Live Marine Debris Map")

m = folium.Map(
    location=[17.5, 70.0],
    zoom_start=5,
    tiles="CartoDB dark_matter"
)

# SATELLITE TILE
if show_tiles and tile_info:

    folium.TileLayer(
        tiles=tile_info["tile_url"],
        attr=tile_info["attribution"],
        name=tile_info["name"],
        overlay=True,
        opacity=0.6
    ).add_to(m)

# HOTSPOTS
for h in hotspots:

    intensity = min(h["fdi"] / 0.15, 1.0)

    colour = "#ff4444" if intensity > 0.6 else "#00e5ff"

    folium.CircleMarker(
        location=[h["lat"], h["lon"]],
        radius=5 + intensity * 6,
        color=colour,
        fill=True,
        fill_color=colour,
        fill_opacity=0.8,
        tooltip=f"FDI {h['fdi']}"
    ).add_to(m)

# ROUTE
if show_route and route["waypoints"]:

    coords = [
        [w["lat"], w["lon"]]
        for w in route["waypoints"]
    ]

    folium.PolyLine(
        coords,
        color="#00e5ff",
        weight=3,
        opacity=0.8
    ).add_to(m)

folium.LayerControl().add_to(m)

st_folium(
    m,
    width="100%",
    height=600
)

# ─────────────────────────────────────────────────────────────
# TABLES
# ─────────────────────────────────────────────────────────────
st.markdown("---")

st.subheader("📋 Route Manifest")

if route["waypoints"]:

    df = pd.DataFrame(route["waypoints"])

    st.dataframe(
        df,
        width="stretch",
        height=300
    )

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")

st.caption(
    "AetherSea-II · Sentinel-2 · Google Earth Engine · NOAA Ocean Currents"
)
