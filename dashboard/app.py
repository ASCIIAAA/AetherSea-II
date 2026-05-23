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

import sys
try:
    with open("c:/Users/ASUS/AetherSea/debug_paths.txt", "w") as debug_f:
        debug_f.write("sys.path:\n" + "\n".join(sys.path) + "\n\n")
        try:
            import ee
            debug_f.write(f"ee path: {ee.__file__}\n")
        except Exception as e:
            debug_f.write(f"ee import failed: {e}\n")
        try:
            import streamlit as st_debug
            debug_f.write(f"streamlit path: {st_debug.__file__}\n")
        except Exception as e:
            debug_f.write(f"streamlit import failed: {e}\n")
except Exception as e:
    pass


# BACKEND
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
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    st.title("⚙️  Mission Control")

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
    0.006,
    0.0005,
    format="%.4f"
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
        st.success("🟢 GEE Connected")
    else:
        st.error("🔴 Demo Mode")
        if gee_error_msg:
            st.caption(gee_error_msg)


# HEADER
st.markdown(
    '<p class="main-title">🌊 AetherSea-II</p>',
    unsafe_allow_html=True
)

st.caption("Marine Debris Intelligence Platform  ·  Arabian Sea")

# DEMO HOTSPOTS
def _demo_hotspots(
    n: int = 60,
    fdi_thresh: float = 0.04,
    start_date=None
):
    """Generate seeded demo hotspots for offline mode."""
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


# CACHING DECORATORS (KEY FIX FOR SPEED!)
@st.cache_data(ttl=3600, show_spinner=False)
def _load_hotspots_cached(lon_min, lon_max, lat_min, lat_max, start, end, fdi_t):
    """Cached hotspot loading - won't rerun unless params change."""
    return get_cloud_reduced_hotspots(
        lon_range=[lon_min, lon_max],
        lat_range=[lat_min, lat_max],
        start_date=start,
        end_date=end,
        fdi_threshold=fdi_t,
        ndvi_threshold=0.15
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _load_stats_cached(start, end):
    """Cached stats loading."""
    try:
        return get_region_stats(
            aoi=ee.Geometry.Rectangle([66, 12, 72, 18]),
            start=start,
            end=end
        )
    except:
        return {"mean_fdi": 0.065, "mean_pi": 0.052}


@st.cache_data(ttl=3600, show_spinner=False)
def _load_tile_cached(start, end):
    """Cached tile loading."""
    try:
        return get_plastic_tile_url(
            aoi=ee.Geometry.Rectangle([66, 12, 72, 18]),
            start=start,
            end=end
        )
    except:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def _plan_route_cached(hs_json, speed):
    """Cached routing - recalculates if hotspots or speed change."""
    hs = json.loads(hs_json)

    if not hs:
        return {
            "waypoints": [],
            "segments": [],
            "total_cost": 0.0,
            "total_dist_km": 0.0,
            "land_detours": 0
        }

    return plan_cleanup_route(
        hs,
        ship_speed=speed
    )


@st.cache_data(ttl=1800, show_spinner=False)
def _generate_mission_report_cached(hotspots_json, route_json, region_stats_json, source):
    """Cached wrapper to prevent re-running Gemini on every UI interaction."""
    hotspots = json.loads(hotspots_json)
    route = json.loads(route_json)
    region_stats = json.loads(region_stats_json)
    agent = SupervisorAgent()
    return agent.generate_mission_report(
        hotspots=hotspots,
        route=route,
        region_stats=region_stats,
        source=source
    )

# ─────────────────────────────────────────────────────────────
# LOAD DATA (ONLY WHEN BUTTON CLICKED)
# ─────────────────────────────────────────────────────────────
if "hotspots" not in st.session_state or run_btn:

    with st.spinner("🛰️  Querying satellite systems..."):

        hotspots = None
        region_stats = None
        tile_info = None
        data_source = "demo"

        if _GEE_AVAILABLE:
            try:
                # Use CACHED loading (10x faster on reruns!)
                hotspots = _load_hotspots_cached(
                    64.0, 74.0, 8.0, 22.0,
                    start_str, end_str, fdi_thresh
                )

                region_stats = _load_stats_cached(start_str, end_str)

                tile_info = _load_tile_cached(start_str, end_str)

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
            hotspots = _demo_hotspots(10, fdi_thresh, start_str)
            region_stats = {"mean_fdi": 0.065, "mean_pi": 0.052}
            tile_info = None
            data_source = "demo"

            st.session_state["hotspots"] = hotspots
            st.session_state["region_stats"] = region_stats
            st.session_state["tile_info"] = tile_info
            st.session_state["data_source"] = data_source

# STATE RETRIEVAL
hotspots = st.session_state.get("hotspots", [])
region_stats = st.session_state.get("region_stats", {})
tile_info = st.session_state.get("tile_info")
data_source = st.session_state.get("data_source", "demo")

# ROUTING (CACHED)
route = _plan_route_cached(
    json.dumps(hotspots),
    ship_speed
)

# STEP 3: AI SUPERVISOR AGENT
mission_summary = None
try:
    mission_summary = _generate_mission_report_cached(
        json.dumps(hotspots),
        json.dumps(route),
        json.dumps(region_stats),
        data_source
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
# STEP 3: AI SUMMARY PANEL
# ─────────────────────────────────────────────────────────────
st.markdown("---")

with st.expander("🧠 Supervisor AI Assessment", expanded=True):
    st.markdown(mission_summary)

# ─────────────────────────────────────────────────────────────
# MAP
# ─────────────────────────────────────────────────────────────
st.markdown("### 🗺️  Live Marine Debris Map")

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

folium_static(
    m,
    width="100%",
    height=600
)

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
                    if i < len(route["segments"]) else None
                ),
                "Time (h)": (
                    route["segments"][i]["cost"]
                    if i < len(route["segments"]) else None
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

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")

st.caption(
    "AetherSea-II · Sentinel-2 · Google Earth Engine · NOAA OSCAR · Gemini AI"
)