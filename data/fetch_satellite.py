# data/fetch_satellite.py

import ee


class SatelliteFetcher:

    def __init__(self, project_name="aethersea"):

        if project_name:
            ee.Initialize(project=project_name)
        else:
            ee.Initialize()

    # ---------------------------------------------------
    # Cloud + Cirrus Masking
    # ---------------------------------------------------

    def mask_clouds(self, image):

        qa = image.select("QA60")

        # Bit masks
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11

        mask = (
            qa.bitwiseAnd(cloud_bit_mask).eq(0)
            .And(
                qa.bitwiseAnd(cirrus_bit_mask).eq(0)
            )
        )

        return (
        image
        .updateMask(mask)
        .divide(10000)
        .copyProperties(
            image,
            image.propertyNames()
        )
    )

    # ---------------------------------------------------
    # Sentinel Query
    # ---------------------------------------------------

    def query_sentinel_image(
        self,
        lon_range,
        lat_range,
        start_date,
        end_date
    ):

        region = ee.Geometry.Rectangle([
            lon_range[0],
            lat_range[0],
            lon_range[1],
            lat_range[1]
        ])

        collection = (
            ee.ImageCollection(
                'COPERNICUS/S2_SR_HARMONIZED'
            )
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(
                ee.Filter.lt(
                    'CLOUDY_PIXEL_PERCENTAGE',
                    15
                )
            )
            .map(self.mask_clouds)
            .sort('CLOUDY_PIXEL_PERCENTAGE')
        )

        image = collection.first()

        if image is None:
            raise ValueError(
                "No cloud-free Sentinel image found."
            )

        return image