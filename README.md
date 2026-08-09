# About the Project: AetherSea

## Inspiration
Marine plastic pollution is one of the most critical environmental threats facing our oceans today[cite: 2]. Millions of tons of plastic debris flood into marine ecosystems every year, causing catastrophic harm to marine biodiversity, ecosystems, and coastal economies[cite: 2, 3]. 

While researching existing technological interventions, we noticed a critical gap in the workflow: **most current solutions focus purely on detection, but detection alone does not clean up the ocean[cite: 2, 3].** Traditional monitoring methods—like manual ship patrols or visual aerial surveys—are slow, expensive, and cover only small coastal areas[cite: 2, 3]. When cleanup vessels *are* deployed, they often navigate blindly along naive Euclidean routes[cite: 2]. Sailing directly against strong ocean currents burns massive amounts of vessel fuel and increases mission durations exponentially[cite: 2, 3].

We were inspired to build **AetherSea** to bridge the gap between space-based earth observation and real-world ocean conservation[cite: 2]. We wanted to create an end-to-end platform that not only detects floating debris in real time from space[cite: 2, 3], but also uses fluid hydrodynamics and optimization algorithms to guide cleanup vessels along the most fuel-efficient, current-assisted recovery paths[cite: 2, 3].

---

## What it does
**AetherSea** is a full-stack marine debris intelligence and physics-aware route optimization platform designed for large-scale ocean monitoring[cite: 2, 3]. 

Key capabilities include:
* **Satellite Debris Detection:** Ingests Sentinel-2 multispectral imagery via Google Earth Engine (GEE) to continuously monitor large ocean basins (such as the Arabian Sea)[cite: 2].
* **Spectral False-Positive Filtering:** Uses custom spectral index calculations—combining the Floating Debris Index (FDI) with Normalized Difference Vegetation Index (NDVI) filtering—to isolate anthropogenic plastic debris while filtering out natural biological matter like Sargassum seaweed[cite: 2, 3].
* **Physics-Aware Hydrodynamic Routing:** Integrates real-time NOAA OSCAR ocean surface current datasets[cite: 2, 3]. Instead of computing straight-line paths, our engine projects ocean velocity vectors onto vessel trajectories to calculate effective travel costs and current-assisted velocity boosts[cite: 1, 2].
* **2-Opt TSP Route Optimization:** Solves a Traveling Salesman Problem (TSP) using 2-Opt heuristics to generate an optimal waypoint sequence that minimizes travel time, fuel burn, and carbon emissions[cite: 2, 3].
* **Generative AI Maritime Briefings:** Leverages a Gemini 2.5 Flash supervisor agent to instantly analyze mission metrics, safety alerts, and navigation manifests, transforming complex mathematical data into plain-language operational briefings for maritime authorities[cite: 1, 2, 3].

---

## How we built it
AetherSea is built on a modular four-tier architecture[cite: 2]:

1. **Data Acquisition & Remote Sensing Layer:** Powered by **Google Earth Engine (GEE)** to process Sentinel-2 harmonized imagery without overloading local compute resources[cite: 2]. Ocean surface current vectors are ingested from **NOAA OSCAR** datasets[cite: 2, 3].
2. **Spectral Processing Engine:** 
   * We calculate the **Floating Debris Index (FDI)** using Sentinel-2's Near-Infrared (NIR), Red Edge, and Short-Wave Infrared (SWIR) bands[cite: 2]:
     $$\text{FDI} = \text{NIR} - \left[ \text{RedEdge} + (\text{SWIR} - \text{RedEdge}) \times \frac{832.8 - 704.1}{1613.7 - 704.1} \right]$$[cite: 2]
   * To eliminate false positives from biological algae, we calculate **NDVI**[cite: 2]:
     $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$[cite: 2]
   * A binary mask is applied where $\text{FDI} > T_{\text{fdi}}$ and $\text{NDVI} < T_{\text{ndvi}}$[cite: 2].
   * GEE's `reduceToVectors()` converts connected plastic pixels into spatial polygons and extracts their centroids[cite: 2, 3].
3. **Physics-Aware Optimization Engine:**
   * Computes a pairwise Haversine distance matrix between all target hotspots[cite: 2].
   * Adjusts vessel speed dynamically based on current vectors[cite: 2]:
     $$V_{\text{effective}} = V_{\text{ship}} + V_{\text{current}}$$[cite: 2]
   * Integrates **GeoPandas** and **Shapely** with Natural Earth landmass polygon shapefiles to inject safety detours around coastline obstacles[cite: 1, 3].
   * Applies a 2-Opt TSP heuristic solver to iteratively optimize the waypoint visit order[cite: 2, 3].
4. **Decision Support & Presentation Layer:** Built with **Streamlit** and **Folium** for an interactive operator dashboard[cite: 2], integrated with the **Gemini 2.5 Flash API** for automated mission briefing generation[cite: 1, 2].

---

## Challenges we ran into
* **Handling Massive Satellite Data Without Node Memory Errors:** Processing high-resolution multispectral imagery over entire ocean basins locally caused severe out-of-memory errors. We solved this by offloading all spatial processing server-side into Google Earth Engine and reducing binary masks to vector centroids (`reduceToVectors()`) at a 5,000m scale, ensuring that only lightweight coordinate payloads ($<5\text{ KB}$) are transferred to our application[cite: 2, 3].
* **Distinguishing Plastics from Natural Ocean Anomalies:** Floating seaweed (such as Sargassum) displays a spectral response similar to floating plastics in near-infrared bands[cite: 2]. We resolved this by implementing dual-index cross-verification: masking out high-NDVI pixels cleanly isolates anthropogenic plastics[cite: 2, 3].
* **Landmass Intersection & Safety Detours:** Standard routing models sometimes generated straight-line vectors crossing nearshore islands or shelf barriers. Integrating GeoPandas spatial boundary checks allowed us to detect land intersections dynamically and inject predefined offshore detour nodes to protect vessel transit[cite: 1, 3].
* **Decoupling Streamlit UI State:** Preventing map interactions from triggering expensive backend re-fetches required designing a non-blocking state management layer inside Streamlit[cite: 3].


---

## What's next for AetherSea
* **Real-Time Ocean Hydrodynamics:** Transitioning from historical/archive NOAA OSCAR feeds to high-frequency live operational current models[cite: 2].
* **ML Debris Density & Microplastic Classification:** Integrating deep learning computer vision models on SAR (Synthetic Aperture Radar) and optical imagery to classify plastic concentration levels and microplastic slick profiles[cite: 2].
* **Autonomous Vessel (ASV) Telemetry:** Integrating direct MAVLink/ROS2 navigation hardware protocols to stream optimized waypoints straight to autonomous surface cleanup drones.
* **Multi-Vessel Fleet Coordination (mVRP):** Expanding our 2-Opt TSP solver into a Multi-Vehicle Routing Problem (mVRP) engine to coordinate multi-ship cleanup fleets simultaneously[cite: 2].

* ---

* ## Take a look

<img width="1920" height="1073" alt="Dashboard" src="https://github.com/user-attachments/assets/da166250-6854-4bc8-9716-d9da0a545599" />

<img width="1920" height="1041" alt="Dashboard2" src="https://github.com/user-attachments/assets/7b5c54d0-790d-464c-a1e5-8f90e3e613b9" />

