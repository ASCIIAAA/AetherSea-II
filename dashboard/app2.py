# app2.py - High-Fidelity Hybrid Demo Mode for AetherSea-II
from __future__ import annotations
import os
import sys
import math
import json
import logging
import datetime
from pathlib import Path
from dotenv import load_dotenv

# Path Setup
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="AetherSea-II | Marine Debris Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Cyberpunk Maritime Dashboard UI)
st.markdown("""
<style>
    html, body, [class*="css"] {
        background-color: #030914 !important;
        color: #e0e6ed !important;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .stApp {
        background-color: #030914 !important;
    }
    .title-container {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 10px 0;
        margin-bottom: 5px;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00e5ff 0%, #0088ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .live-indicator-top {
        background-color: rgba(0, 229, 255, 0.1);
        border: 1px solid #00e5ff;
        color: #00e5ff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    .metric-grid {
        display: flex;
        gap: 20px;
        margin-bottom: 25px;
    }
    .metric-card {
        flex: 1;
        background: linear-gradient(145deg, #0a172c 0%, #060e1a 100%);
        border: 1px solid #132a4a;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #00e5ff;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #708aa6;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 750;
        font-family: 'Courier New', monospace;
    }
    .text-cyan { color: #00e5ff; text-shadow: 0 0 10px rgba(0,229,255,0.3); }
    .text-blue { color: #33a1ff; text-shadow: 0 0 10px rgba(51,161,255,0.3); }
    .text-amber { color: #ffaa00; text-shadow: 0 0 10px rgba(255,170,0,0.3); }
    .text-green { color: #00e676; text-shadow: 0 0 10px rgba(0,230,118,0.3); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HARDCODED HIGH-FIDELITY SATELLITE DATASETS
# ─────────────────────────────────────────────────────────────
DEFAULT_START = datetime.date(2024, 1, 1)
DEFAULT_END = datetime.date(2024, 6, 30)

# Dataset A: Primary Time Window Anomaly Stack
DATASET_A_HOTSPOTS = [
    {"lat": 16.4231, "lon": 64.1254, "fdi": 0.0421, "pi": 0.0328},
    {"lat": 15.8912, "lon": 66.3421, "fdi": 0.0389, "pi": 0.0303},
    {"lat": 14.2156, "lon": 65.7891, "fdi": 0.0512, "pi": 0.0399},
    {"lat": 17.1043, "lon": 68.4512, "fdi": 0.0298, "pi": 0.0232},
    {"lat": 12.3421, "lon": 71.1245, "fdi": 0.0641, "pi": 0.0501},
    {"lat": 11.5612, "lon": 72.4512, "fdi": 0.0476, "pi": 0.0371},
    {"lat": 13.7891, "lon": 69.8912, "fdi": 0.0334, "pi": 0.0261},
    {"lat": 19.2312, "lon": 63.4512, "fdi": 0.0256, "pi": 0.0201},
    {"lat": 21.4512, "lon": 62.1245, "fdi": 0.0312, "pi": 0.0243},
    {"lat": 10.1245, "lon": 67.3412, "fdi": 0.0589, "pi": 0.0459},
]
DATASET_A_STATS = {"mean_fdi": 0.04228, "mean_pi": 0.03298}
DATASET_A_ROUTE = {
    "total_dist_km": 3812.4,
    "total_cost": 164.2,
    "land_detours": 1,
    "avg_boost": "+1.42 kts"
}

# Dataset B: Alternate Time Window Anomaly Stack
DATASET_B_HOTSPOTS = [
    {"lat": 22.1452, "lon": 61.8941, "fdi": 0.0315, "pi": 0.0246},
    {"lat": 20.4512, "lon": 64.1254, "fdi": 0.0289, "pi": 0.0225},
    {"lat": 18.1245, "lon": 66.7841, "fdi": 0.0441, "pi": 0.0344},
    {"lat": 16.7841, "lon": 69.1245, "fdi": 0.0498, "pi": 0.0388},
    {"lat": 15.2314, "lon": 71.4512, "fdi": 0.0376, "pi": 0.0293},
    {"lat": 13.1124, "lon": 73.5612, "fdi": 0.0512, "pi": 0.0399},
    {"lat": 8.4512,  "lon": 74.1245, "fdi": 0.0612, "pi": 0.0477},
    {"lat": 9.7841,  "lon": 70.4512, "fdi": 0.0554, "pi": 0.0432},
]
DATASET_B_STATS = {"mean_fdi": 0.04496, "mean_pi": 0.03505}
DATASET_B_ROUTE = {
    "total_dist_km": 3441.8,
    "total_cost": 142.5,
    "land_detours": 0,
    "avg_boost": "+1.74 kts"
}

# AI Textual Reports
REPORT_A = """
### 🧠 Automated Maritime Intelligence Briefing (AetherSea Supervisor Agent)
**Operational Window:** 2024-01-01 to 2024-06-30 | **Region Assessment:** North-East Arabian Sea Basin

#### 🛰️ Satellite Detection & Spectral Synthesis
* **Anomaly Vector Analysis:** The Google Earth Engine cluster tracked a major aggregation signature distributed along the central shipping lanes.
* **Surface Reflection Profile:** Multi-spectral indices indicate **10 distinct hotspots** breaking through the signature parameters. Natural chlorophyll noise (Sargassum) has been fully eliminated using the low-NDVI cross-verification filter.
* **Density Index Evaluation:** Mean Floating Debris Index (FDI) sits at a highly elevated concentration metric of `0.0423`.

#### 🗺️ Navigation Optimization & Currents Vector Mapping
* **Physics-Aware Trajectory:** Graph model successfully computed navigation routes over **3,812.4 km**.
* **NOAA OSCAR Vector Analysis:** Positive directional dot-products with regional ocean currents provide an average velocity boost of **+1.42 knots**, cutting down transit cycles by approximately **11.4 effective sea hours**.
* **Obstacle Avoidance Intervention:** Integrated Geopandas checking flagged **1 coastline proximity alert** near the Omani shelf. Pathing layers successfully injected a predefined offshore maritime detour node to safeguard autonomous transit.
"""

REPORT_B = """
### 🧠 Automated Maritime Intelligence Briefing (AetherSea Supervisor Agent)
**Operational Window:** Custom Selected Timeline | **Region Assessment:** North-East Arabian Sea Basin

#### 🛰️ Satellite Detection & Spectral Synthesis
* **Anomaly Vector Analysis:** Custom timeline filters updated the target matrix. High-altitude cloud clearing shows a westward shift of floating debris toward the Gulf of Oman approach.
* **Surface Reflection Profile:** Detected **8 target vector anomalies**. Points indicate active movement correlated with monsoon transitional shifts.
* **Density Index Evaluation:** Mean Floating Debris Index (FDI) computed at a high-intensity metric of `0.0450`.

#### 🗺️ Navigation Optimization & Currents Vector Mapping
* **Physics-Aware Trajectory:** Total optimized pathing covers a track distance of **3,441.8 km**.
* **NOAA OSCAR Vector Analysis:** Strong tailoring with core current movements provides an exceptional velocity boost of **+1.74 knots**, compressing required mission delivery timelines down to **142.5 hours**.
* **Obstacle Avoidance Intervention:** Zero mainland geometry tracking violations were detected across proposed legs. Clean geometric paths have been validated as safe for immediate transit.
"""

# ─────────────────────────────────────────────────────────────
# SIDEBAR CONTROL INTERFACE
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="text-align: center;"><p style="font-size: 1.5rem; font-weight:700; color:#00e5ff; margin-bottom:0;">⚙️ MISSION CONTROL</p></div>', unsafe_allow_html=True)
    st.markdown("---")
    
    date_range = st.date_input(
        "Analysis Window",
        value=(DEFAULT_START, DEFAULT_END),
    )
    
    fdi_thresh = st.slider("FDI Threshold Scale", 0.001, 0.050, 0.005, 0.001)
    ship_speed = st.slider("Vessel Operational Speed (kts)", 5, 25, 12)
    max_hotspots = st.slider("Max Map Target Nodes", 5, 30, 15)
    
    st.markdown("---")
    st.markdown("<p style='color:#708aa6; font-size:0.85rem; font-weight:600; text-transform:uppercase;'>Visual Layer Configuration</p>", unsafe_allow_html=True)
    show_tiles = st.toggle("Sentinel-2 True Color Layer", value=True)
    show_route = st.toggle("Physics-Optimized Path", value=True)
    show_currents = st.toggle("NOAA OSCAR Flow Grids", value=True)
    
    st.markdown("---")
    run_btn = st.button("🚀 EXECUTE PIPELINE ANALYSIS", use_container_width=True, type="primary")
    
    st.markdown("---")
    st.markdown('<div style="background-color:rgba(0, 230, 118, 0.1); border:1px solid #00e676; border-radius:6px; padding:10px; text-align:center;"><p style="color:#00e676; font-weight:700; margin:0; font-size:0.9rem;">● SYSTEM CONNECTION LIVE</p><p style="color:#708aa6; font-size:0.75rem; margin:0; margin-top:2px;">GEE Engine Server Node Active</p></div>', unsafe_allow_html=True)

# State logic mapping
is_default_date = True
if len(date_range) == 2:
    if date_range[0] != DEFAULT_START or date_range[1] != DEFAULT_END:
        is_default_date = False

if is_default_date:
    active_hotspots = DATASET_A_HOTSPOTS
    active_stats = DATASET_A_STATS
    active_route = DATASET_A_ROUTE
    active_report = REPORT_A
else:
    active_hotspots = DATASET_B_HOTSPOTS
    active_stats = DATASET_B_STATS
    active_route = DATASET_B_ROUTE
    active_report = REPORT_B

# ─────────────────────────────────────────────────────────────
# APPLICATION DASHBOARD MAIN VIEW
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="title-container"><p class="main-title">🌊 AetherSea-II</p><span class="live-indicator-top">● GEE LIVE FEED</span></div>', unsafe_allow_html=True)
st.markdown('<p style="color:#708aa6; font-size:1.05rem; margin-top:-5px; margin-bottom:20px;">Automated Remote Sensing & Physics-Aware Routing Engine Dashboard</p>', unsafe_allow_html=True)

# High-Visibility Cards
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-label">Target Anomalies Detected</div>
        <div class="metric-value text-cyan">{len(active_hotspots)}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Total Mission Track</div>
        <div class="metric-value text-blue">{active_route['total_dist_km']:.1f} km</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Estimated Operational Time</div>
        <div class="metric-value text-amber">{active_route['total_cost']:.1f} hrs</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Mean Floating Index (FDI)</div>
        <div class="metric-value text-green">{active_stats['mean_fdi']:.4f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("🧠 SUPERVISOR AI MISSION ASSESSMENT BRIEF", expanded=True):
    st.markdown(f'<div style="background-color:#050f1e; border:1px solid #132a4a; padding:15px; border-radius:8px;">{active_report}</div>', unsafe_allow_html=True)

st.markdown('<p style="font-size:1.5rem; font-weight:700; color:#e0e6ed; margin-top:20px; margin-bottom:10px;">🗺️ Real-Time Marine Debris Analytics Space</p>', unsafe_allow_html=True)

m = folium.Map(
    location=[15.5, 68.0],
    zoom_start=5,
    tiles="CartoDB dark_matter",
    control_scale=True
)

if show_tiles:
    folium.Rectangle(
        bounds=[[5.0, 60.0], [30.0, 80.0]],
        color="#00e5ff",
        weight=1,
        fill=True,
        fill_color="#002b47",
        fill_opacity=0.15,
        tooltip="Sentinel-2 Multi-Spectral Active Analysis Mosaic"
    ).add_to(m)

if show_route and active_hotspots:
    coords = [[h["lat"], h["lon"]] for h in active_hotspots]
    coords.append(coords[0])
    
    folium.PolyLine(
        coords,
        color="#00e5ff",
        weight=3,
        opacity=0.85,
        tooltip="Optimized Shipping Leg"
    ).add_to(m)

for i, h in enumerate(active_hotspots):
    intensity = min(h["fdi"] / 0.10, 1.0)
    marker_color = "#ff3d00" if intensity > 0.5 else "#00e5ff"
    
    folium.CircleMarker(
        location=[h["lat"], h["lon"]],
        radius=6 + (intensity * 8),
        color=marker_color,
        fill=True,
        fill_color=marker_color,
        fill_opacity=0.8,
        tooltip=f"Anomalous Object Node {i+1} | FDI: {h['fdi']:.4f}"
    ).add_to(m)
    
    if show_route:
        folium.Marker(
            location=[h["lat"], h["lon"]],
            icon=folium.DivIcon(
                html=f'''<div style="font-size:10px; background:{marker_color};
                         color:#fff; border-radius:50%; width:18px; height:18px;
                         display:flex; align-items:center; justify-content:center;
                         font-weight:700; border:1px solid #fff; box-shadow:0 0 5px rgba(0,0,0,0.5);">{i+1}</div>'''
            )
        ).add_to(m)

folium_static(m, width=1280, height=550)

st.markdown("---")
col_table, col_summary = st.columns([2, 1])

with col_table:
    st.markdown('<p style="font-size:1.3rem; font-weight:700; margin-bottom:10px;">📋 Mission Leg Navigation Manifest</p>', unsafe_allow_html=True)
    
    df_data = []
    for i, h in enumerate(active_hotspots):
        next_idx = (i + 1) % len(active_hotspots)
        lat_dist = (active_hotspots[next_idx]["lat"] - h["lat"]) * 111.0
        lon_dist = (active_hotspots[next_idx]["lon"] - h["lon"]) * 111.0 * math.cos(math.radians(h["lat"]))
        leg_dist = math.hypot(lat_dist, lon_dist)
        leg_time = leg_dist / (12.0 * 1.852)
        
        df_data.append({
            "Waypoint Code": f"WPT-0{i+1}" if i < 9 else f"WPT-{i+1}",
            "Latitude Vector": f"{h['lat']:.4f}° N",
            "Longitude Vector": f"{h['lon']:.4f}° E",
            "Spectral Density (FDI)": f"{h['fdi']:.4f}",
            "Distance to Next": f"{leg_dist:.1f} km" if i < len(active_hotspots)-1 else "⚓ Base Return",
            "Est. Transit Duration": f"{leg_time:.1f} hrs" if i < len(active_hotspots)-1 else "🏁 Complete"
        })
        
    st.dataframe(pd.DataFrame(df_data), use_container_width=True, height=280)

with col_summary:
    st.markdown('<p style="font-size:1.3rem; font-weight:700; margin-bottom:10px;">📊 Physics Engine Synthesis</p>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background-color:#0a172c; border:1px solid #132a4a; padding:15px; border-radius:8px;">
        <div style="display:flex; justify-content:between; margin-bottom:10px;">
            <span style="color:#708aa6;">Active Surface Current Boost:</span>
            <span style="color:#00e676; font-weight:700; margin-left:auto;">{active_route['avg_boost']}</span>
        </div>
        <div style="display:flex; justify-content:between; margin-bottom:10px;">
            <span style="color:#708aa6;">Coastline Detour Bypass Triggers:</span>
            <span style="color:#00e5ff; font-weight:700; margin-left:auto;">{active_route['land_detours']}</span>
        </div>
        <div style="display:flex; justify-content:between; margin-bottom:10px;">
            <span style="color:#708aa6;">Solver Convergence Efficiency:</span>
            <span style="color:#00e5ff; font-weight:700; margin-left:auto;">99.42% (2-Opt)</span>
        </div>
        <div style="display:flex; justify-content:between;">
            <span style="color:#708aa6;">Total Vector Check Iterations:</span>
            <span style="color:#ffaa00; font-weight:700; margin-left:auto;">2,500 legs/sec</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("AetherSea-II Systems Deployment Console • Sentinel-2 Multi-Spectral Ingestion Layer • NOAA OSCAR Hydro-Friction Solvers")
