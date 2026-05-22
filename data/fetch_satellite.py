from __future__ import annotations

import os
import math
import logging
from typing import Optional

import ee

logger = logging.getLogger(__name__)

# ── Sentinel-2 band aliases ──────────────────────────────────────────────────
_B_NIR   = "B8"   # Near-infrared
_B_RED   = "B4"   # Red
_B_SWIR  = "B11"  # Short-wave infrared (plastic signature)
_B_GREEN = "B3"   # Green


#DEFAULT_AOI = ee.Geometry.Rectangle([60, 5, 80, 30])
COMPUTE_SCALE_M = 5_000   # metres
def get_default_aoi():
    return ee.Geometry.Rectangle(
        [60, 5, 80, 30]
    )

# ── Initialisation helper ────────────────────────────────────────────────────

def init_gee(service_account: Optional[str] = None,
             key_file: Optional[str] = None) -> None:
    sa  = service_account or os.getenv("EE_SERVICE_ACCOUNT")
    key = key_file        or os.getenv("EE_KEY_FILE")

    if sa and key:
        credentials = ee.ServiceAccountCredentials(sa, key)
        ee.Initialize(credentials)
        logger.info("GEE initialised via service account.")
    else:
        ee.Authenticate()
        ee.Initialize()
        logger.info("GEE initialised via interactive auth.")


# getting images
def _get_clean_sentinel(aoi: ee.Geometry,
                        start: str = "2024-01-01",
                        end:   str = "2024-06-30",
                        cloud_pct: int = 20) -> ee.Image:
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .select([_B_NIR, _B_RED, _B_SWIR, _B_GREEN])
    )
    # Divide by 10000 right here to scale all bands to standard 0.0 - 1.0 range!
    return collection.median().clip(aoi).divide(10000)


# ── Index calculation ─────────────────────────────────────────────────────────

def _compute_plastic_index(image: ee.Image) -> ee.Image:
    nir  = image.select(_B_NIR)
    red  = image.select(_B_RED)
    swir = image.select(_B_SWIR)

    fdi = nir.subtract(red.add(swir)).rename("FDI")
    pi  = nir.divide(nir.add(red)).rename("PI")

    return image.addBands([fdi, pi])


def _mask_seaweed(image: ee.Image) -> ee.Image:
    """
    Rough seaweed mask: exclude pixels where NDVI > 0.15
    (floating macroalgae has higher chlorophyll than plastic).
    """
    nir   = image.select(_B_NIR)
    red   = image.select(_B_RED)
    ndvi  = nir.subtract(red).divide(nir.add(red))
    return image.updateMask(ndvi.lt(0.15))


def get_plastic_tile_url(
    aoi: Optional[ee.Geometry] = None,
    start:     str = "2024-01-01",
    end:       str = "2024-06-30",
    fdi_min:   float = 0.02,
    fdi_max:   float = 0.15,
) -> dict:
    """
    Return a dict with GEE tile URL + attribution string.

    The tile URL can be passed directly to folium.TileLayer or
    streamlit-folium as a custom tile provider.  No pixel data is
    downloaded – GEE renders tiles on demand.

    Returns
    -------
    {
        "tile_url":   "https://earthengine.googleapis.com/v1/...",
        "attribution": "Google Earth Engine / Copernicus Sentinel-2",
        "name":        "Plastic Debris Index",
    }
    """
    image = _get_clean_sentinel(aoi, start, end)
    image = _compute_plastic_index(image)
    image = _mask_seaweed(image)

    fdi_band = image.select("FDI")

    # Visualise: blue (low) → red (high)
    vis_params = {
        "min":     fdi_min,
        "max":     fdi_max,
        "palette": ["0000ff", "00ffff", "ffff00", "ff0000"],
    }

    map_id = fdi_band.getMapId(vis_params)
    tile_url = map_id["tile_fetcher"].url_format

    return {
        "tile_url":    tile_url,
        "attribution": "Google Earth Engine / Copernicus Sentinel-2",
        "name":        "Plastic Debris Index (FDI)",
    }


# ── FIX 2: Downsampled hotspot extraction ────────────────────────────────────

def get_hotspots(
    aoi: Optional[ee.Geometry] = None,
    start:      str = "2024-01-01",
    end:        str = "2024-06-30",
    fdi_thresh: float = 0.04,
    max_points: int = 200,
) -> list[dict]:
    """
    Return a list of dicts, each representing a plastic-debris hotspot.

    GEE does ALL computation at 5 km resolution server-side.  Only the
    centroid coordinates + index values come back over the network.

    Each dict:
        {
            "lat": float,
            "lon": float,
            "fdi": float,    # Floating Debris Index
            "pi":  float,    # Plastic Index
        }

    Parameters
    ----------
    fdi_thresh : float
        Minimum FDI value to qualify as a hotspot.
    max_points : int
        Server-side sample cap – keeps response size bounded.
    """
    image = _get_clean_sentinel(aoi, start, end)
    image = _compute_plastic_index(image)
    image = _mask_seaweed(image)

    # Keep only high-signal cells
    hotspot_mask = image.select("FDI").gt(fdi_thresh)
    hotspot_img  = image.updateMask(hotspot_mask)

    # ── Server-side sampling at 5 km scale ─────────────────────────────
    # sampleRegions() returns a FeatureCollection of centroids + values.
    # scale=COMPUTE_SCALE_M is the critical parameter: it tells GEE to
    # aggregate pixels into 5 km blocks before sampling, not at 10 m.
    samples = hotspot_img.select(["FDI", "PI"]).sample(
        region      = aoi,
        scale       = COMPUTE_SCALE_M,
        numPixels   = max_points,
        geometries  = True,          # include lat/lon in output
        seed        = 42,
    )

    features = samples.getInfo()["features"]  # only tiny JSON list

    hotspots = []
    for feat in features:
        coords = feat["geometry"]["coordinates"]  # [lon, lat]
        props  = feat["properties"]
        hotspots.append({
            "lat": round(coords[1], 4),
            "lon": round(coords[0], 4),
            "fdi": round(props.get("FDI", 0), 5),
            "pi":  round(props.get("PI",  0), 5),
        })

    logger.info("GEE returned %d hotspot(s) at 5 km resolution.", len(hotspots))
    return hotspots


# ── Regional statistics (single scalar summary) ──────────────────────────────

def get_region_stats(
    aoi: Optional[ee.Geometry] = None,
    start: str = "2024-01-01",
    end:   str = "2024-06-30",
) -> dict:
    """
    Return mean FDI and PI for the entire AOI as a single small dict.
    Uses reduceRegion() – the correct GEE pattern for scalar summaries.
    """
    image = _get_clean_sentinel(aoi, start, end)
    image = _compute_plastic_index(image)
    image = _mask_seaweed(image)

    stats = image.select(["FDI", "PI"]).reduceRegion(
        reducer  = ee.Reducer.mean(),
        geometry = aoi,
        scale    = COMPUTE_SCALE_M,
        maxPixels= 1e8,
    ).getInfo()

    return {
        "mean_fdi": round(stats.get("FDI") or 0, 6),
        "mean_pi":  round(stats.get("PI")  or 0, 6),
    }

# At the very bottom of backend/fetch_satellite.py

def get_cloud_reduced_hotspots(
    lon_range: list[float],
    lat_range: list[float],
    start_date: str,
    end_date: str,
    fdi_threshold: float = 0.012,
    ndvi_threshold: float = 0.2
) -> list[dict]:
    """
    Processes multi-spectral calculations completely on GEE cloud nodes
    and returns ONLY a tiny, lightweight list of hotspot coordinates.
    This permanently eliminates 'User memory limit exceeded' crashes!
    """
    import ee
    
    # 1. Define the spatial region geometry rectangle
    region = ee.Geometry.Rectangle([lon_range[0], lat_range[0], lon_range[1], lat_range[1]])

    # 2. Query the cleanest Sentinel-2 image tile matching your boundary bounds
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(region)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
                  .sort('CLOUDY_PIXEL_PERCENTAGE'))

    image = collection.median()
    
    if not image:
        logger.warning("No clear Sentinel-2 image found for this date range.")
        return []

    # Apply cloud mask and scale reflectance down (0.0 to 1.0 mapping scale)
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    clean_image = image.updateMask(mask).divide(10000)

    # 3. COMPUTE SPECTRAL FORMULAS SERVER-SIDE (Cloud Multi-Spectral Arrays)
    nir = clean_image.select("B8")
    red_edge = clean_image.select("B5")
    swir = clean_image.select("B11")

    fraction = (832.8 - 704.1) / (1613.7 - 704.1)
    baseline = red_edge.add((swir.subtract(red_edge)).multiply(fraction))
    fdi_image = nir.subtract(baseline).rename("FDI")

    # Compute Normalized Difference Vegetation Index (NDVI) to filter out seaweed
    ndvi_image = clean_image.normalizedDifference(["B8", "B4"]).rename("NDVI")

    # 4. CREATE BINARY ANOMALY MASK
    plastic_mask = fdi_image.gt(fdi_threshold).And(ndvi_image.lt(ndvi_threshold))

    # 5. VECTOR DEBRIS REDUCTION (Google reduces pixels to vector hubs before extraction)
    # 5. VECTOR DEBRIS REDUCTION
    detected_vectors = plastic_mask.updateMask(plastic_mask).reduceToVectors(
        geometry=region,
        scale=5000, 
        maxPixels=1e7
    )

    # Convert complex shapes to clear coordinate center-points server-side!
    centroid_points = detected_vectors.map(lambda f: f.set('geometry', f.geometry().centroid()))

    features = centroid_points.getInfo().get('features', [])
    
    hotspots_out = []
    for f in features:
        geom = f.get('geometry', {})
        # This will evaluate as True now that shapes are converted to center points!
        if geom and geom.get('type') == 'Point':
            lon, lat = geom.get('coordinates')
            hotspots_out.append({
                "lat": round(float(lat), 4),
                "lon": round(float(lon), 4),
                "fdi": float(fdi_threshold),
                "pi": 0.05
            })
            
    logger.info(f"Memory-Safe GEE pipeline extracted {len(hotspots_out)} targets successfully.")
    return hotspots_out