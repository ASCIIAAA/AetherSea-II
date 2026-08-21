# 🌊 AetherSea: Autonomous Marine Habitat Protection System

<p align="center">
  <b>An AI-powered predictive marine conservation platform for detecting, tracking, and intercepting floating plastic debris before it reaches vulnerable ecosystems.</b>
</p>


## Overview

Marine plastic pollution is one of the biggest threats to coastal biodiversity. Current cleanup operations are largely reactive, discovering debris only after it reaches shorelines and sensitive habitats.

**AetherSea** introduces a predictive, autonomous approach by combining:

- 🛰️ Multispectral satellite observation
- 🌊 Ocean current hydrodynamics
- 🤖 Multi-agent AI reasoning
- 🗺️ Geospatial machine learning
- 🚢 Autonomous fleet route optimization

The platform detects floating plastic clusters, predicts their movement using ocean currents, evaluates ecological risk, and generates optimized interception strategies for autonomous cleanup vessels.

---

# 🏗️ System Architecture

AetherSea follows a closed-loop intelligence pipeline:
Satellite Observation
        ↓
Floating Debris Detection (FDI)
        ↓
Spatial Clustering (DBSCAN)
        ↓
Ocean Drift Forecasting (HYCOM)
        ↓
Habitat Threat Assessment
        ↓
Autonomous Vessel Routing (A*)
        ↓
Multi-Agent Mission Generation



---

# 🛰️ Technology Stack

| Layer | Technology |
|------|------------|
| Satellite Data | ESA Sentinel-2 MSI |
| Ocean Data | HYCOM Global Ocean Model |
| Spectral Analysis | Floating Debris Index (FDI) |
| Machine Learning | DBSCAN Clustering |
| Geospatial Processing | GeoPandas, Shapely |
| Route Optimization | A* Search Algorithm |
| AI Agents | Gemini-powered Multi-Agent Architecture |
| Backend | Python |
| Dashboard | Streamlit + Interactive Maps |
| Visualization | Plotly, Leaflet |


---

# 🛰️ How We Built It


## 1. Multi-Spectral Floating Debris Detection

AetherSea uses Sentinel-2 multispectral imagery at 10m resolution to identify floating polymer signatures over seawater.

The Floating Debris Index (FDI) is calculated as:

$$
FDI = R_{NIR} -
[
R_{RED}
+
(R_{SWIR1}-R_{RED})
\times
\frac{\lambda_{NIR}-\lambda_{RED}}
{\lambda_{SWIR1}-\lambda_{RED}}
\times 10
]
$$


Where:

- $R_{RED}$ is Band 4 reflectance ($\lambda_{RED}=665 nm$)
- $R_{NIR}$ is Band 8 reflectance ($\lambda_{NIR}=842 nm$)
- $R_{SWIR1}$ is Band 11 reflectance ($\lambda_{SWIR1}=1610 nm$)


This allows identification of anomalous spectral signatures associated with floating debris.


---

# 2. Spatial Clustering & Convex Hull Delineation

Detected debris pixels are grouped using:

**Density-Based Spatial Clustering of Applications with Noise (DBSCAN)**


Parameters:
ε = 0.04°
MinPts = 4


Each detected debris region is represented using Convex Hull geometry:


$$
Hull(C_k)=
\{
\sum_{i=1}^{|C_k|}\alpha_i x_i :
\alpha_i \geq 0,
\sum_{i=1}^{|C_k|}\alpha_i = 1
\}
$$


This converts scattered debris detections into meaningful spatial clusters.


---

# 3. Lagrangian Hydrodynamic Drift Prediction

Floating debris movement is influenced by ocean currents.

AetherSea integrates velocity fields:

$(u,v)$

from the **Hybrid Coordinate Ocean Model (HYCOM)**.


Forward trajectory prediction is performed using:


$$
x(t+\Delta t)=
x(t)+
\int_t^{t+\Delta t}
u(x,\tau)d\tau
$$


$$
y(t+\Delta t)=
y(t)+
\int_t^{t+\Delta t}
v(x,\tau)d\tau
$$


The system forecasts possible debris movement over:

- 24-hour
- 48-hour
- 72-hour


time windows.


---

# 4. Habitat Threat Scoring (HTS)

Not all debris poses the same ecological risk.

AetherSea introduces a **Habitat Threat Score (HTS)** to prioritize debris approaching sensitive marine regions.


$$
HTS=
\overline{FDI}
\times
S_{habitat}
\times
(
\frac{R_{sanctuary}\times2.5}
{max(1.0,D_{proj})}
)
\times
\gamma_{convergence}
$$


The score considers:

- Floating debris intensity
- Habitat sensitivity
- Distance from protected ecosystems
- Current convergence patterns


This enables intelligent prioritization of cleanup operations.


---

# 5. Autonomous Fleet Route Optimization

To dispatch Autonomous Surface Vessels (ASVs), AetherSea implements a grid-based A* search algorithm.


The optimization function:

$$
f(n)=g(n)+h(n)
$$


The routing engine considers:

- Water-only navigation paths
- Coastal barriers
- Shallow regions
- Optimal interception distance


---

#  Multi-Agent AI Architecture

AetherSea uses specialized AI agents that collaborate to generate autonomous mission strategies.


## Sentinel Observation Agent

Responsibilities:

- Processes satellite observations
- Extracts Floating Debris Index information
- Identifies debris clusters


---

## Hydrodynamic Drift Agent

Responsibilities:

- Processes HYCOM current data
- Simulates debris movement
- Predicts future positions


---

## Sanctuary Risk Agent

Responsibilities:

- Evaluates habitat vulnerability
- Calculates Habitat Threat Scores
- Assigns priority levels


---

## Fleet Tactical Dispatch Agent

Responsibilities:

- Generates vessel routes
- Performs A* optimization
- Creates mission instructions


---

# AetherSea Mission OS Dashboard

The interactive dashboard provides:


## Sanctuary Tactical Radar

Features:

✅ Marine protected area monitoring  
✅ Debris cluster visualization  
✅ Predicted drift trajectories  
✅ Threat prioritization  


---

## Sentinel-2 FDI Analytics Studio

Features:

✅ Pixel-level reflectance analysis  
✅ Spectral signature comparison  
✅ DBSCAN cluster analysis  


---

##  HYCOM Hydrodynamic Engine

Features:

✅ Ocean velocity visualization  
✅ Current vector analysis  
✅ 72-hour drift forecasting  


---

##  Autonomous Multi-Agent War Room

Features:

✅ Agent communication  
✅ Automated mission synthesis  
✅ Structured JSON mission generation  


# Challenges Solved


## Spectral False Positives

Problem:

Ocean glare, clouds, and foam can resemble floating debris.


Solution:

- SWIR band filtering
- Signal-to-noise analysis
- DBSCAN noise removal


---

## Satellite Processing Latency

Problem:

Live raster processing caused delays.


Solution:

- Vectorized NumPy processing
- Optimized spatial operations
- Precomputed spatial structures


---

## Coastal Navigation Constraints

Problem:

Straight-line vessel paths crossed land regions.


Solution:

- Water-mask constrained A* routing
- Collision-free navigation planning


---

## Ecological Prioritization

Problem:

Raw debris coordinates lacked environmental context.


Solution:

Created the Habitat Threat Score (HTS) framework.


---

# Datasets & Attribution


## Sentinel-2 MSI

European Space Agency (ESA)

Used for:

- Multispectral satellite imagery
- Floating debris detection


---

## HYCOM

Hybrid Coordinate Ocean Model

Used for:

- Ocean current velocity fields
- Drift simulation


---

## Natural Earth

Used for:

- Coastal boundaries
- Marine geographic features


---

## Google Gemini

Used for:

- Multi-agent AI orchestration
- Mission generation


---

# Future Improvements


- Real-time Sentinel-2 API integration
- Deep learning-based debris segmentation
- Live autonomous vessel telemetry
- Integration with marine conservation organizations
- Field validation using real-world observations


---

# Vision

AetherSea aims to transform marine conservation from a reactive cleanup process into a predictive intelligence system — enabling early detection, smarter decisions, and autonomous protection of vulnerable ocean ecosystems.