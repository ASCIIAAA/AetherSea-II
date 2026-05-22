"""
fetch_satellite.py (STEP 1 FIXED VERSION)
==========================================
KEY CHANGE:
  The get_cloud_reduced_hotspots() function now properly extracts actual
  FDI and PI values from the computed raster instead of using hardcoded
  placeholder values.

  Before:
    "fdi": round(fdi_threshold, 5),  # just echoes the threshold
    "pi": 0.05,                       # hardcoded dummy

  After:
    "fdi": round(props.get("FDI", 0), 5),  # actual computed FDI value
    "pi": round(props.get("PI", 0), 5),    # actual computed PI value
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import ee

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Sentinel-2 Bands
# ─────────────────────────────────────────────────────────────
_B_NIR = "B8"
_B_RED = "B4"
_B_SWIR = "B11"
_B_GREEN = "B3"

COMPUTE_SCALE_M = 5000


# ─────────────────────────────────────────────────────────────
# Default AOI
# ─────────────────────────────────────────────────────────────
def get_default_aoi():
    return ee.Geometry.Rectangle([60, 5, 80, 30])


# ─────────────────────────────────────────────────────────────
# GEE INIT
# ─────────────────────────────────────────────────────────────
def init_gee(
    service_account: Optional[str] = None,
    key_file: Optional[str] = None,
):
    try:
        ee.Initialize()
        return
    except Exception:
        pass

    sa = service_account or os.getenv("EE_SERVICE_ACCOUNT")
    key = key_file or os.getenv("EE_KEY_FILE")

    if sa and key:
        credentials = ee.ServiceAccountCredentials(sa, key)
        ee.Initialize(credentials)
        logger.info("GEE initialised via service account.")
    else:
        ee.Authenticate()
        ee.Initialize()
        logger.info("GEE initialised via interactive auth.")


# ─────────────────────────────────────────────────────────────
# CLOUD MASK
# ─────────────────────────────────────────────────────────────
def _mask_s2_clouds(image):
    qa = image.select("QA60").toInt()

    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    mask = (
        qa.bitwiseAnd(cloud_bit_mask)
        .eq(0)
        .And(
            qa.bitwiseAnd(cirrus_bit_mask).eq(0)
        )
    )

    return image.updateMask(mask)


# ─────────────────────────────────────────────────────────────
# CLEAN SENTINEL IMAGE
# ─────────────────────────────────────────────────────────────
def _get_clean_sentinel(
    aoi: Optional[ee.Geometry] = None,
    start: str = "2024-01-01",
    end: str = "2024-06-30",
    cloud_pct: int = 20,
):
    if aoi is None:
        aoi = get_default_aoi()

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(_mask_s2_clouds)
    )

    image = (
        collection.median()
        .clip(aoi)
        .select([_B_NIR, _B_RED, _B_SWIR, _B_GREEN])
        .divide(10000)
    )

    return image


# ─────────────────────────────────────────────────────────────
# PLASTIC INDEX
# ─────────────────────────────────────────────────────────────
def _compute_plastic_index(image):
    nir = image.select(_B_NIR)
    red = image.select(_B_RED)
    swir = image.select(_B_SWIR)

    fdi = nir.subtract(
        red.add(swir)
    ).rename("FDI")

    pi = nir.divide(
        nir.add(red)
    ).rename("PI")

    return image.addBands([fdi, pi])


# ─────────────────────────────────────────────────────────────
# SEAWEED MASK
# ─────────────────────────────────────────────────────────────
def _mask_seaweed(image):
    ndvi = image.normalizedDifference(
        [_B_NIR, _B_RED]
    )

    return image.updateMask(ndvi.lt(0.15))


# ─────────────────────────────────────────────────────────────
# TILE URL
# ─────────────────────────────────────────────────────────────
def get_plastic_tile_url(
    aoi: Optional[ee.Geometry] = None,
    start: str = "2024-01-01",
    end: str = "2024-06-30",
    fdi_min: float = 0.02,
    fdi_max: float = 0.15,
):
    if aoi is None:
        aoi = get_default_aoi()

    image = _get_clean_sentinel(aoi, start, end)
    image = _compute_plastic_index(image)
    image = _mask_seaweed(image)

    fdi_band = image.select("FDI")

    vis_params = {
        "min": fdi_min,
        "max": fdi_max,
        "palette": [
            "0000ff",
            "00ffff",
            "ffff00",
            "ff0000",
        ],
    }

    map_id = fdi_band.getMapId(vis_params)

    return {
        "tile_url": map_id["tile_fetcher"].url_format,
        "attribution": "Google Earth Engine / Sentinel-2",
        "name": "Plastic Debris Index",
    }


# ─────────────────────────────────────────────────────────────
# HOTSPOTS
# ─────────────────────────────────────────────────────────────
def get_hotspots(
    aoi: Optional[ee.Geometry] = None,
    start: str = "2024-01-01",
    end: str = "2024-06-30",
    fdi_thresh: float = 0.04,
    max_points: int = 200,
):
    if aoi is None:
        aoi = get_default_aoi()

    image = _get_clean_sentinel(aoi, start, end)
    image = _compute_plastic_index(image)
    image = _mask_seaweed(image)

    hotspot_mask = image.select("FDI").gt(fdi_thresh)

    hotspot_img = image.updateMask(hotspot_mask)

    samples = hotspot_img.select(
        ["FDI", "PI"]
    ).sample(
        region=aoi,
        scale=COMPUTE_SCALE_M,
        numPixels=max_points,
        geometries=True,
        seed=42,
    )

    features = samples.getInfo()["features"]

    hotspots = []

    for feat in features:
        coords = feat["geometry"]["coordinates"]
        props = feat["properties"]

        hotspots.append({
            "lat": round(coords[1], 4),
            "lon": round(coords[0], 4),
            "fdi": round(props.get("FDI", 0), 5),
            "pi": round(props.get("PI", 0), 5),
        })

    logger.info(
        "GEE returned %d hotspots",
        len(hotspots)
    )

    return hotspots


# ─────────────────────────────────────────────────────────────
# REGION STATS
# ─────────────────────────────────────────────────────────────
def get_region_stats(
    aoi: Optional[ee.Geometry] = None,
    start: str = "2024-01-01",
    end: str = "2024-06-30",
):
    if aoi is None:
        aoi = get_default_aoi()

    image = _get_clean_sentinel(aoi, start, end)
    image = _compute_plastic_index(image)
    image = _mask_seaweed(image)

    stats = image.select(
        ["FDI", "PI"]
    ).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=COMPUTE_SCALE_M,
        maxPixels=1e8,
    ).getInfo()

    return {
        "mean_fdi": round(stats.get("FDI") or 0, 6),
        "mean_pi": round(stats.get("PI") or 0, 6),
    }


# CLOUD REDUCED HOTSPOTS (STEP 1 IMPROVED)
def get_cloud_reduced_hotspots(
    lon_range, lat_range, start_date, end_date, fdi_threshold=0.012, ndvi_threshold=0.2
):
    region = ee.Geometry.Rectangle([lon_range[0], lat_range[0], lon_range[1], lat_range[1]])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 15)) # Strictly drop cloudy tiles early
        .map(_mask_s2_clouds)
    )
    
    base_image = collection.median().clip(region).divide(10000)
    nir = base_image.select("B8")
    red = base_image.select("B4")
    swir = base_image.select("B11")

    fdi = nir.subtract(red.add(swir)).rename("FDI")
    pi = nir.divide(nir.add(red)).rename("PI")
    ndvi = base_image.normalizedDifference(["B8", "B4"]).rename("NDVI")

    # Plastic mask: High FDI + Low NDVI
    plastic_mask = fdi.gt(fdi_threshold).And(ndvi.lt(ndvi_threshold))

    # Build composite for sampling
    analysis_image = base_image.addBands([fdi, pi, ndvi]).updateMask(plastic_mask)

    # Sample targets server-side
    samples = analysis_image.select(["FDI", "PI"]).sample(
        region=region,
        scale=COMPUTE_SCALE_M,
        numPixels=150,  # Lowered slightly for lightning-fast demo responses
        geometries=True,
        seed=42,
    )

    features = samples.getInfo()["features"]
    hotspots = []

    for f in features:
        coords = f["geometry"]["coordinates"]
        props = f["properties"]
        hotspots.append({
            "lat": round(coords[1], 4),
            "lon": round(coords[0], 4),
            "fdi": round(props.get("FDI", 0), 5),
            "pi": round(props.get("PI", 0), 5),
        })

    return hotspots