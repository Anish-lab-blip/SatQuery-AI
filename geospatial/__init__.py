"""SatQuery AI geospatial: CRS comparison, affine transforms, alignment, overlap."""

from geospatial.crs import compare_crs, describe_crs, is_metric, parse_crs
from geospatial.transform import (
    GeoBounds,
    PixelWindow,
    geo_to_pixel,
    iou,
    normalized_to_pixel,
    pixel_to_geo,
    pixel_to_normalized,
)

__all__ = [
    "compare_crs",
    "describe_crs",
    "is_metric",
    "parse_crs",
    "GeoBounds",
    "PixelWindow",
    "geo_to_pixel",
    "pixel_to_geo",
    "pixel_to_normalized",
    "normalized_to_pixel",
    "iou",
]