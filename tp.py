
import ee
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Sentinel-2 band definitions (L2A Harmonized)
SENTINEL2_BANDS = {
    'B2': 'Blue (490nm)',
    'B3': 'Green (560nm)',
    'B4': 'Red (665nm)',
    'B5': 'Red Edge (705nm)',
    'B6': 'Red Edge (740nm)',
    'B7': 'Red Edge (783nm)',
    'B8': 'NIR (842nm)',
    'B8A': 'Narrow NIR (865nm)',
    'B11': 'SWIR (1610nm)',
    'B12': 'SWIR (2190nm)',
    'QA60': 'Cloud mask (bitmask)',
    'SCL': 'Scene classification mask'
}

# Cloud mask bit positions in QA60
CLOUD_MASK_BITS = {
    'opaque_clouds': 10,      # Bit 10 = opaque clouds
    'cirrus_clouds': 11       # Bit 11 = cirrus clouds
}

# FDI thresholds for different debris types
FDI_SENSITIVITY = {
    'high_confidence': 0.040,    # Dense plastic mats/islands
    'moderate': 0.025,           # Dispersed plastic patches
    'sensitive': 0.015,          # Small debris clusters
    'very_sensitive': 0.010      # Individual plastic items (many false positives)
}

# ============================================================================
# AUTHENTICATION & INITIALIZATION
# ============================================================================

def authenticate_gee(service_account_path: str) -> bool:
    try:
        # Validate service account file exists
        with open(service_account_path, 'r') as f:
            credentials_data = json.load(f)
        
        # Initialize with service account
        credentials = ee.ServiceAccountCredentials(
            email=credentials_data.get('client_email'),
            key_data=credentials_data.get('private_key')
        )
        
        ee.Initialize(credentials)
        
        logger.info("✓ Google Earth Engine authenticated successfully")
        logger.info(f"  Service Account: {credentials_data.get('client_email')}")
        
        return True
        
    except FileNotFoundError:
        logger.error(f"✗ Service account file not found: {service_account_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"✗ Invalid JSON in service account file")
        raise
    except ee.EEException as e:
        logger.error(f"✗ GEE Authentication failed: {str(e)}")
        raise


# ============================================================================
# CLOUD FILTERING UTILITIES
# ============================================================================

def create_cloud_mask(image: ee.Image) -> ee.Image:

    qa60 = image.select('QA60')
    
    # Create mask for opaque clouds (bit 10)
    opaque_mask = qa60.bitwiseAnd(1 << CLOUD_MASK_BITS['opaque_clouds'])
    
    # Create mask for cirrus clouds (bit 11)
    cirrus_mask = qa60.bitwiseAnd(1 << CLOUD_MASK_BITS['cirrus_clouds'])
    
    # Combine: cloud_mask = 0 where clear, 1 where cloudy
    cloud_mask = opaque_mask.Or(cirrus_mask)
    
    # Invert: 1 where clear, 0 where cloudy
    clear_mask = cloud_mask.eq(0)
    
    return clear_mask


def calculate_cloud_percentage(image: ee.Image, geometry: ee.Geometry) -> float:
  
    cloud_mask = create_cloud_mask(image)
    
    cloud_fraction = cloud_mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=10,
        maxPixels=1e8
    ).getInfo()
    
    # Cloud fraction is (1 - clear_mask), so we subtract from 1
    cloud_percent = (1 - cloud_fraction.get('QA60', 1.0)) * 100
    
    return cloud_percent


# ============================================================================
# SPECTRAL INDEX CALCULATION
# ============================================================================

def calculate_floating_debris_index(image: ee.Image) -> ee.Image:
    # Select bands (already in reflectance scale 0-10000)
    nir = image.select('B8').float()
    red_edge = image.select('B5').float()
    swir = image.select('B11').float()
    
    # Avoid division by zero
    sum_bands = nir.add(red_edge)
    
    # Calculate normalized difference
    norm_diff = nir.subtract(red_edge).divide(sum_bands.max(1))
    
    # Add SWIR component (normalized to 0-1 range)
    swir_normalized = swir.divide(10000)
    
    # Combined FDI
    fdi = norm_diff.add(swir_normalized)
    
    return fdi.rename('FDI')


def calculate_ndvi(image: ee.Image) -> ee.Image:
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return ndvi


def create_plastic_mask(
    image: ee.Image,
    fdi_threshold: float = 0.025,
    ndvi_threshold: float = 0.3
) -> ee.Image:
    fdi = calculate_floating_debris_index(image)
    ndvi = calculate_ndvi(image)
    cloud_mask = create_cloud_mask(image)
    
    # Plastic detection logic
    high_fdi = fdi.gt(fdi_threshold)
    low_ndvi = ndvi.lt(ndvi_threshold)
    is_clear = cloud_mask.eq(1)  # 1 = clear pixel
    
    # Combine all conditions
    plastic_mask = high_fdi.And(low_ndvi).And(is_clear)
    
    return plastic_mask.rename('PLASTIC')


# MEDIAN COMPOSITE GENERATION (THE FIX)

def generate_median_composite(
    collection: ee.ImageCollection,
    geometry: ee.Geometry,
    start_date: str,
    end_date: str
) -> ee.Image:
    logger.info(f"📊 Generating median composite ({start_date} to {end_date})")
    
    # Filter collection
    filtered = collection \
        .filterDate(start_date, end_date) \
        .filterBounds(geometry) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
    
    # Check if we have any images
    image_count = filtered.size().getInfo()
    logger.info(f"   Found {image_count} images with <15% cloud coverage")
    
    if image_count == 0:
        logger.warning("   ⚠ No cloud-free images found! Relaxing threshold to 30%")
        filtered = collection \
            .filterDate(start_date, end_date) \
            .filterBounds(geometry) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
        image_count = filtered.size().getInfo()
        logger.info(f"   Found {image_count} images with <30% cloud coverage")
    
    if image_count == 0:
        raise ValueError(f"No cloud-free Sentinel-2 images available for {start_date} to {end_date}")
    
    # Create per-image cloud masks and apply them
    def mask_clouds(img):
        cloud_mask = create_cloud_mask(img)
        return img.updateMask(cloud_mask)

    masked_collection = filtered.map(mask_clouds)
    
    # Generate median composite
    composite = masked_collection.median()
    
    logger.info(f"   ✓ Median composite generated from {image_count} images")
    
    return composite


def generate_median_composite_with_interpolation(
    collection: ee.ImageCollection,
    geometry: ee.Geometry,
    start_date: str,
    end_date: str
) -> ee.Image:
    logger.info(f"🔄 Generating interpolated composite (advanced mode)")
    
    # Parse dates
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    
    # Collect weekly composites
    week_composites = []
    current = start
    
    while current < end:
        week_end = current + timedelta(days=7)
        if week_end > end:
            week_end = end
        
        week_composite = generate_median_composite(
            collection,
            geometry,
            current.isoformat().split('T')[0],
            week_end.isoformat().split('T')[0]
        )
        
        week_composites.append(week_composite)
        logger.info(f"   Week {current.date()}: composite added")
        
        current = week_end
    
    # Stack all weekly composites and take final median
    if len(week_composites) == 0:
        raise ValueError("No weekly composites generated")
    elif len(week_composites) == 1:
        return week_composites[0]
    else:
        # Create ImageCollection from weekly composites
        composite_collection = ee.ImageCollection(week_composites)
        final_composite = composite_collection.median()
        logger.info(f"   ✓ Final composite from {len(week_composites)} weeks")
        return final_composite


def get_cloud_reduced_hotspots(
    start_date: str,
    end_date: str,
    fdi_threshold: float,
    region_bounds: Dict[str, float],
    service_account_path: str,
    min_cluster_area: float = 0.5
) -> List[List[float]]:
    logger.info("=" * 70)
    logger.info("HOTSPOT DETECTION: Cloud-Reduced Median Composite Pipeline")
    logger.info("=" * 70)
    
    # Authenticate
    authenticate_gee(service_account_path)
    
    # Validate date range
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")
    
    if start >= end:
        raise ValueError(f"Start date must be before end date")
    
    days_span = (end - start).days
    if days_span > 365:
        logger.warning(f"⚠ Large date range ({days_span} days) may be computationally intensive")
    
    logger.info(f"📅 Date Range: {start_date} to {end_date} ({days_span} days)")
    logger.info(f"🎯 FDI Threshold: {fdi_threshold}")
    logger.info(f"🗺️  Region: {region_bounds['min_lon']}°E-{region_bounds['max_lon']}°E, "
                f"{region_bounds['min_lat']}°N-{region_bounds['max_lat']}°N")
    
    # Define region geometry
    roi = ee.Geometry.BBox(
        region_bounds['min_lon'],
        region_bounds['min_lat'],
        region_bounds['max_lon'],
        region_bounds['max_lat']
    )
    
    logger.info(f"📡 Loading Sentinel-2 L2A Harmonized collection...")
    
    # Load Sentinel-2 collection
    sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    
    try:
        # ✅ FIX STEP 1: Generate median composite (not .first())
        logger.info(f"🔄 Generating median composite across date range...")
        composite = generate_median_composite(
            sentinel2,
            roi,
            start_date,
            end_date
        )
        
        # ✅ FIX STEP 2: Calculate indices
        logger.info(f"📊 Calculating spectral indices...")
        plastic_mask = create_plastic_mask(
            composite,
            fdi_threshold=fdi_threshold,
            ndvi_threshold=0.3
        )
        
        # ✅ FIX STEP 3: Convert to vectors
        logger.info(f"🎯 Converting pixels to hotspot vectors...")
        vectors = plastic_mask.reduceToVectors(
            geometry=roi,
            scale=5000,  # 5km resolution for memory efficiency
            maxPixels=1e8,
            geometryType='centroid'
        )
        
        # ✅ FIX STEP 4: Extract coordinates
        logger.info(f"📍 Extracting hotspot coordinates...")
        geometry_info = vectors.geometry().getInfo()
        
        hotspots = []
        
        if geometry_info and geometry_info.get('type') == 'FeatureCollection':
            features = geometry_info.get('features', [])
            logger.info(f"   Found {len(features)} hotspot features")
            
            for feature in features:
                try:
                    geom = feature.get('geometry', {})
                    
                    if geom.get('type') == 'Point':
                        coords = geom.get('coordinates', [])
                        if len(coords) == 2:
                            lon, lat = coords
                            hotspots.append([float(lon), float(lat)])
                    
                    elif geom.get('type') == 'MultiPoint':
                        coords_list = geom.get('coordinates', [])
                        for coords in coords_list:
                            if len(coords) == 2:
                                lon, lat = coords
                                hotspots.append([float(lon), float(lat)])
                
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(f"   ⚠ Skipped malformed feature: {e}")
                    continue
        
        logger.info("=" * 70)
        logger.info(f"✓ DETECTION COMPLETE: Found {len(hotspots)} debris hotspots")
        logger.info("=" * 70)
        
        return hotspots
    
    except ee.EEException as e:
        logger.error(f"✗ Google Earth Engine error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"✗ Unexpected error during hotspot detection: {str(e)}")
        raise


def get_fdi_statistics(
    start_date: str,
    end_date: str,
    region_bounds: Dict[str, float],
    service_account_path: str
) -> Dict[str, float]:
    logger.info("📊 Calculating FDI statistics for threshold calibration...")
    
    authenticate_gee(service_account_path)
    
    roi = ee.Geometry.BBox(
        region_bounds['min_lon'],
        region_bounds['min_lat'],
        region_bounds['max_lon'],
        region_bounds['max_lat']
    )
    
    sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    composite = generate_median_composite(sentinel2, roi, start_date, end_date)
    
    fdi = calculate_floating_debris_index(composite)
    
    stats = fdi.reduceRegion(
        reducer=ee.Reducer.percentile([5, 25, 50, 75, 95]),
        geometry=roi,
        scale=100,
        maxPixels=1e8
    ).getInfo()
    
    results = {
        'p5': float(stats.get('FDI_p5', 0)),
        'p25': float(stats.get('FDI_p25', 0)),
        'p50': float(stats.get('FDI_p50', 0)),
        'p75': float(stats.get('FDI_p75', 0)),
        'p95': float(stats.get('FDI_p95', 0))
    }
    
    logger.info(f"   FDI Distribution:")
    logger.info(f"   5th percentile:  {results['p5']:.4f}")
    logger.info(f"   25th percentile: {results['p25']:.4f}")
    logger.info(f"   Median (50th):   {results['p50']:.4f}")
    logger.info(f"   75th percentile: {results['p75']:.4f}")
    logger.info(f"   95th percentile: {results['p95']:.4f}")
    
    return results

if __name__ == "__main__":
    # Example usage
    region = {
        'min_lon': 60,
        'max_lon': 80,
        'min_lat': 5,
        'max_lat': 30
    }
    
    try:
        # Get FDI stats first
        stats = get_fdi_statistics(
            "2024-01-01",
            "2024-01-31",
            region,
            ".streamlit/secrets.toml"
        )
        
        # Detect hotspots using calibrated threshold
        hotspots = get_cloud_reduced_hotspots(
            "2024-01-01",
            "2024-01-31",
            fdi_threshold=0.025,
            region_bounds=region,
            service_account_path=".streamlit/secrets.toml"
        )
        
        print(f"\n✓ Detected {len(hotspots)} hotspots:")
        for i, (lon, lat) in enumerate(hotspots[:5], 1):
            print(f"   {i}. {lon:.2f}°E, {lat:.2f}°N")
    
    except Exception as e:
        logger.error(f"Script failed: {e}")