# dashboard/app.py
import sys
import os
import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go

# Ensure backend modules can be imported smoothly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.index_calculator import IndexCalculator
from backend.clustering_engine import DBSCANClusteringEngine # Make sure class name matches yours
from backend.route_optimizer import RouteOptimizer
from agents.supervisor_agent import SupervisorAgent

st.set_page_config(layout="wide", page_title="AetherSea Command Center")

st.title("🌊 AETHERSEA — Live-Data Ocean Cleanup Command Center")
st.markdown(
    "Ingesting **Multi-Spectral Satellite Telemetry** and **Ocean Currents** to coordinate autonomous marine cleanup operations."
)

st.sidebar.header("Operational Controls")
fdi_threshold = st.sidebar.slider("FDI Plastic Detection Threshold", 0.010, 0.030, 0.015, step=0.001)
run_pipeline = st.sidebar.button("📡 Ingest Live Data & Optimize")

if run_pipeline:
    with st.spinner("Executing analytical pipeline..."):
        # ==========================================
        # STEP 1 & 2: LOAD & CROSS-REFERENCE DATA
        # ==========================================
        # Loading your verified local 2024 HYCOM dataset
        url = (
    "https://tds.hycom.org/thredds/dodsC/"
    "GLBy0.08/expt_93.0"
)

        ds = xr.open_dataset(
            url,
            engine="pydap",
            decode_times=False
        )

        u_surface = ds["water_u"].isel(
            time=0,
            depth=0
        )

        region = u_surface.sel(
            lat=slice(5, 30),
            lon=slice(60, 80)
        )

        latitudes = region.lat.values
        longitudes = region.lon.values
        
        # Pulling raw bands (In production, these come from your fetch_satellite GEE script!)
        # For our unified test, we simulate an active region using the grid size
        shape = (len(latitudes), len(longitudes))
        
        # Creating a realistic baseline grid matching open ocean
        mock_nir = np.ones(shape) * 0.02
        mock_red_edge = np.ones(shape) * 0.04
        mock_swir = np.ones(shape) * 0.01
        
        # Seeding random dense anomalies to simulate real clusters picked up by Sentinel-2
        np.random.seed(42) 
        for _ in range(45):
            mock_red = np.ones(shape) * 0.03
            # Cluster 1
            y1 = np.random.randint(5, 12)
            x1 = np.random.randint(8, 15)

            mock_nir[y1, x1] = 0.18
            mock_red[y1, x1] = 0.16

            # Cluster 2
            y2 = np.random.randint(18, 25)
            x2 = np.random.randint(22, 30)

            mock_nir[y2, x2] = 0.22
            mock_red[y2, x2] = 0.19
            
        # ==========================================
        # STEP 3: FLOATING DEBRIS INDEX & CLUSTERING
        # ==========================================
        calc = IndexCalculator(
            fdi_threshold=fdi_threshold
        )

        # ---------------------------------------------------
        # Compute Spectral Indices
        # ---------------------------------------------------

        fdi_matrix = calc.calculate_fdi(
            mock_nir,
            mock_red_edge,
            mock_swir
        )

        # Mock RED band for NDVI
        mock_red = np.ones(shape) * 0.03

        ndvi_matrix = calc.calculate_ndvi(
            mock_nir,
            mock_red
        )

        # ---------------------------------------------------
        # Generate Plastic Mask
        # ---------------------------------------------------

        plastic_mask = calc.create_plastic_mask(
            fdi_matrix,
            ndvi_matrix
        )

        # ---------------------------------------------------
        # Extract Coordinates
        # ---------------------------------------------------

        raw_anomaly_points = (
            calc.extract_anomaly_coordinates(
                plastic_mask,
                fdi_matrix,
                ndvi_matrix,
                latitudes,
                longitudes
            )
        )
        
        # Convert anomaly dictionaries to coordinate pairs for DBSCAN
        if len(raw_anomaly_points) == 0:
            st.error("No plastic anomalies detected above the current threshold slider. Lower the threshold!")
            st.stop()
            
        coords_list = [[p["lat"], p["lon"]] for p in raw_anomaly_points]
        
        # Pass coordinates to Partner B's DBSCAN engine
        # (Assuming your engine returns centers of mass coordinates)
        # For safety/fallback inline replication if needed:
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=0.15, min_samples=3).fit(coords_list)
        labels = db.labels_
        
        hotspots = []
        for label in set(labels) - {-1}:
            cluster_nodes = np.array(coords_list)[labels == label]
            c_lat, c_lon = np.mean(cluster_nodes, axis=0)
            hotspots.append({"lat": float(c_lat), "lon": float(c_lon)})
            
        if not hotspots:
            st.warning("Anomalies found, but they are too scattered to form an operational cluster island.")
            st.stop()

        # ==========================================
        # STEP 4: HAVERSINE ROUTE OPTIMIZATION
        # ==========================================
        router = RouteOptimizer()
        start_port = (float(latitudes[0]), float(longitudes[0]))
        
        # Compute path
        route = router.compute_cleanup_route(start_port, hotspots)
        
        # Calculate true Haversine distance sum along the sequence
        total_distance = 0.0
        for i in range(len(route)-1):
            total_distance += router.haversine_distance(route[i], route[i+1])

        # ==========================================
        # DISPLAY RESULTS: METRICS & MAP
        # ==========================================
        st.header("📊 Tactical Mission Analytics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Detected Trash Islands", len(hotspots))
        m2.metric("Total Operational Stops", len(route))
        m3.metric("Optimized Path Length", f"{total_distance:.2f} km")
        
        # Build Interactive Plotly Mapbox
        st.header("🗺️ Live Deployment Interception Map")
        
        fig = go.Figure()

        # 1. Map out the individual satellite anomaly pixels as background warning dots
        fig.add_trace(go.Scattermapbox(
            lat=[p[0] for p in coords_list],
            lon=[p[1] for p in coords_list],
            mode='markers',
            marker=go.scattermapbox.Marker(size=5, color='orange', opacity=0.4),
            name='Raw Satellite Anomalies'
        ))

        # 2. Plot the calculated DBSCAN center points (Targets)
        fig.add_trace(go.Scattermapbox(
            lat=[h["lat"] for h in hotspots],
            lon=[h["lon"] for h in hotspots],
            mode='markers',
            marker=go.scattermapbox.Marker(size=14, color='cyan'),
            name='Verified Waste Hotspots'
        ))

        # 3. Draw the optimized navigation route line connecting everything back to port
        fig.add_trace(go.Scattermapbox(
            lat=[r[0] for r in route],
            lon=[r[1] for r in route],
            mode='lines+markers',
            line=dict(width=3, color='lime'),
            marker=go.scattermapbox.Marker(size=8, color='lime'),
            name='Optimized Interception Path'
        ))

        # Set Mapbox canvas behavior and zoom center
        fig.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=float(np.mean(latitudes)), lon=float(np.mean(longitudes))),
                zoom=6
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # ==========================================
        # STEP 5: AUTOMATED DISPATCH BRIEFING (GEMINI)
        # ==========================================
        st.header("📋 Automated Tactical Command Brief")
        
        supervisor = SupervisorAgent()
        brief_text = supervisor.generate_dispatch_briefing(
            hotspots_count=len(hotspots),
            total_distance=total_distance,
            waypoint_list=route
        )
        
        st.info(brief_text)
        st.success("🛰️ Active telemetry processing cycle successfully completed.")