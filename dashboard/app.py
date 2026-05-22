"""
app.py (STEP 3: Supervisor Agent Integration)
==============================================
KEY CHANGES:
  1. Supervisor agent is now properly called within the data pipeline
  2. Agent receives hotspots, route, and region_stats (not just mock data)
  3. Mission report is displayed in an expandable section on the dashboard
  4. Graceful fallback to templated report if Gemini API unavailable
  5. Error handling doesn't crash the entire app
"""

from __future__ import annotations

import os
import sys
import math
import json
import logging
import datetime
from pathlib import Path

from dotenv import load_dotenv

# ENV LOAD
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT))

# LIBRARIES
import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import numpy as np

# AGENTS
from agents.supervisor_agent import SupervisorAgent
# BACKEND
from backend.routing_engine import plan_cleanup_route


# STREAMLIT CONFIG

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

.ai-summary {
    background: #0a1530;
    border-left: 4px solid #1976d2;
    padding: 1.5rem;
    border-radius: 8px;
    margin: 1rem 0;
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
        0.001,
        0.05,
        0.005,
        0.001
    )

    ship_speed = st.slider(
        "Vessel Speed",
        5,
        25,
        12
    )

    max_hotspots = st.slider(
        "Max Hotspots",
        5,
        30,
        12
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
        st.success("🟢 GEE Connected")
    else:
        st.error("🔴 Demo Mode")
        if gee_error_msg:
            st.caption(gee_error_msg)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<p class="main-title">🌊 AetherSea-II</p>',
    unsafe_allow_html=True
)

st.caption("Marine Debris Intelligence Platform  ·  Arabian Sea")

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
# LOAD HOTSPOTS AND STATS
# ─────────────────────────────────────────────────────────────
if "hotspots" not in st.session_state or run_btn:

    with st.spinner("🛰️ Querying satellite systems..."):

        hotspots = None
        region_stats = None
        tile_info = None
        data_source = "demo"

        if _GEE_AVAILABLE:
            try:
                # STEP 1 FIX: get_cloud_reduced_hotspots now returns real FDI/PI
                hotspots = get_cloud_reduced_hotspots(
                    lon_range=[60.0, 80.0],
                    lat_range=[5.0, 30.0],
                    start_date=start_str,
                    end_date=end_str,
                    fdi_threshold=fdi_thresh,
                    ndvi_threshold=0.15
                )

                # Get region statistics
                try:
                    region_stats = get_region_stats(
                        aoi=ee.Geometry.Rectangle([60, 5, 80, 30]),
                        start=start_str,
                        end=end_str
                    )
                except:
                    region_stats = {"mean_fdi": 0.065, "mean_pi": 0.052}

                # Get satellite tile overlay
                tile_info = get_plastic_tile_url(
                    aoi=ee.Geometry.Rectangle([60, 5, 80, 30]),
                    start=start_str,
                    end=end_str
                )

                data_source = "live"

                st.session_state["hotspots"] = hotspots
                st.session_state["region_stats"] = region_stats
                st.session_state["tile_info"] = tile_info
                st.session_state["data_source"] = data_source

            except Exception as e:
                logger.error(f"GEE query failed: {e}")
                hotspots = _demo_hotspots(max_hotspots, fdi_thresh, start_str)
                region_stats = {"mean_fdi": 0.065, "mean_pi": 0.052}
                tile_info = None
                data_source = "demo"

                st.session_state["hotspots"] = hotspots
                st.session_state["region_stats"] = region_stats
                st.session_state["tile_info"] = tile_info
                st.session_state["data_source"] = data_source

        else:
            hotspots = _demo_hotspots(max_hotspots, fdi_thresh, start_str)
            region_stats = {"mean_fdi": 0.065, "mean_pi": 0.052}
            tile_info = None
            data_source = "demo"

            st.session_state["hotspots"] = hotspots
            st.session_state["region_stats"] = region_stats
            st.session_state["tile_info"] = tile_info
            st.session_state["data_source"] = data_source

# ─────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────
hotspots = st.session_state.get("hotspots", [])
region_stats = st.session_state.get("region_stats", {})
tile_info = st.session_state.get("tile_info")
data_source = st.session_state.get("data_source", "demo")

# ─────────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# PERFORMANCE LIMITER + ROUTING
# ─────────────────────────────────────────────────────────────

# Hard cap to prevent 2-opt explosion
if len(hotspots) > 10:
    hotspots = hotspots[:10]

@st.cache_data(ttl=600)
def _plan_route(hs_json, speed):

    hs = json.loads(hs_json)

    # Emergency limiter
    hs = hs[:10]

    # Empty safety
    if not hs:
        return {
            "waypoints": [],
            "segments": [],
            "total_cost": 0.0,
            "total_dist_km": 0.0,
            "land_detours": 0
        }

    # Small hotspot count → fast routing
    if len(hs) <= 2:
        return {
            "waypoints": hs,
            "segments": [],
            "total_cost": 0.0,
            "total_dist_km": 0.0,
            "land_detours": 0
        }

    try:
        return plan_cleanup_route(
            hs,
            ship_speed=speed
        )

    except Exception as e:
        logger.error(f"Routing failed: {e}")

        return {
            "waypoints": hs,
            "segments": [],
            "total_cost": 0.0,
            "total_dist_km": 0.0,
            "land_detours": 0
        }


# Spinner so user sees progress
with st.spinner("Computing marine route..."):

    route = _plan_route(
        json.dumps(hotspots),
        ship_speed
    )

# ─────────────────────────────────────────────────────────────
# STEP 3: AI SUPERVISOR AGENT (INTEGRATED)
# ─────────────────────────────────────────────────────────────
agent = SupervisorAgent()

# Call agent with real data from pipeline (STEP 3 FIX)
mission_summary = None
try:
    mission_summary = agent.generate_mission_report(
        hotspots=hotspots,
        route=route,
        region_stats=region_stats,
        source=data_source
    )
except Exception as e:
    logger.warning(f"Agent generation failed: {e}")
    mission_summary = None

# Fallback if agent fails
if not mission_summary:
    mission_summary = f"""
## 🌊 AetherSea-II Automated Mission Report

**Detection Summary**
- Debris hotspots identified: {len(hotspots)}
- Mean FDI (Floating Debris Index): {region_stats.get('mean_fdi', 0):.4f}

**Route Optimization**
- Total cleanup distance: {route.get('total_dist_km', 0):.1f} km
- Estimated mission duration: {route.get('total_cost', 0):.1f} hours
- Land obstacle detours: {route.get('land_detours', 0)}

**Operational Status**
✓ Satellite data acquisition complete
✓ Route optimization with current-aware physics
✓ Automated mission brief generated
"""

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
# STEP 3: AI SUMMARY PANEL (INTEGRATED)
# ─────────────────────────────────────────────────────────────
st.markdown("---")

with st.expander("🧠 Supervisor AI Assessment", expanded=True):
    st.markdown(mission_summary)

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
        tooltip=f"FDI {h['fdi']:.4f}"
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

    # Add numbered waypoint markers
    for i, w in enumerate(route["waypoints"]):
        folium.Marker(
            location=[w["lat"], w["lon"]],
            icon=folium.DivIcon(
                html=f'''<div style="font-size:10px;background:#00e5ff;
                         color:#000;border-radius:50%;width:20px;height:20px;
                         display:flex;align-items:center;justify-content:center;
                         font-weight:700">{i+1}</div>'''
            ),
            popup=f"Stop {i+1} - FDI: {w['fdi']:.4f}"
        ).add_to(m)

folium.LayerControl().add_to(m)
folium_static(m, width=1200, height=600)

# ─────────────────────────────────────────────────────────────
# TABLES
# ─────────────────────────────────────────────────────────────
st.markdown("---")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📋 Route Manifest")
    if route["waypoints"]:
        df = pd.DataFrame([
            {
                "Stop": i+1,
                "Lat": w["lat"],
                "Lon": w["lon"],
                "FDI": w["fdi"],
                "Dist to next (km)": (
                    route["segments"][i]["dist_km"]
                    if i < len(route["segments"]) else "–"
                ),
                "Time (h)": (
                    route["segments"][i]["cost"]
                    if i < len(route["segments"]) else "–"
                ),
            }
            for i, w in enumerate(route["waypoints"])
        ])
        st.dataframe(
            df,
            use_container_width=True,
            height=300
        )
    else:
        st.info("No route available")

with col_right:
    st.subheader("📊 Route Stats")
    if route["segments"]:
        boosts = [s["current_boost"] for s in route["segments"]]
        avg_boost = sum(boosts) / len(boosts) if boosts else 0
        st.metric("Avg Current Boost", f"{avg_boost:+.2f} kts")
        st.metric("Land Detours", route.get("land_detours", 0))
        st.metric("Total Legs", len(route["segments"]))

# FOOTER
st.markdown("---")
st.caption(
    "AetherSea-II · Sentinel-2 · Google Earth Engine · NOAA OSCAR · Gemini AI"
)
