"""Sentinel-2 L2A discovery and raster loading via Microsoft Planetary Computer STAC.

Requires internet access. Public STAC availability and data coverage can vary.
AOI is supplied as a WGS84 bounding box: min_lon, min_lat, max_lon, max_lat.
"""
from datetime import datetime
import numpy as np
import planetary_computer
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from pystac_client import Client


STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def search_sentinel2_items(bbox, start_date, end_date, max_cloud=30, limit=30):
    if len(bbox) != 4:
        raise ValueError("Bounding box must be min_lon, min_lat, max_lon, max_lat.")
    min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox]
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        raise ValueError("Invalid WGS84 bounding box.")
    datetime_range = f"{start_date}/{end_date}"
    catalog = Client.open(STAC_URL)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[min_lon, min_lat, max_lon, max_lat],
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": float(max_cloud)}},
        max_items=int(limit),
    )
    items = list(search.items())
    items.sort(key=lambda item: (item.properties.get("eo:cloud_cover", 100), item.datetime or datetime.min))
    return items


def _read_band_to_grid(asset_href, dst_crs, dst_transform, width, height, resampling=Resampling.bilinear):
    out = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(asset_href) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )
    return out


def load_sentinel_item(item, resolution=20):
    """Load B02/B03/B04/B08 and SCL onto one projected grid.

    Returns RGB uint8, NDVI float32, valid-data mask, and raster metadata.
    The scene is cropped only by its source asset footprint here; users should
    inspect the returned footprint and cloud/quality mask before interpreting.
    """
    signed = planetary_computer.sign(item)
    assets = signed.assets
    needed = ["B02", "B03", "B04", "B08", "SCL"]
    missing = [name for name in needed if name not in assets]
    if missing:
        raise ValueError(f"Sentinel-2 item is missing assets: {missing}")

    # Use the red band as the target grid. Reproject each other band to this grid.
    red_href = assets["B04"].href
    with rasterio.open(red_href) as src:
        dst_crs = src.crs
        src_transform = src.transform
        src_width, src_height = src.width, src.height
        src_bounds = src.bounds
        # Keep native grid for first implementation; selected resolution is metadata only.
        dst_transform = src_transform
        width, height = src_width, src_height

    bands = {}
    for name in ["B02", "B03", "B04", "B08"]:
        bands[name] = _read_band_to_grid(
            assets[name].href, dst_crs, dst_transform, width, height,
            Resampling.bilinear
        )
    scl = _read_band_to_grid(
        assets["SCL"].href, dst_crs, dst_transform, width, height,
        Resampling.nearest
    ).astype(np.uint8)

    # Sentinel-2 L2A SCL classes 3 (cloud shadow), 8/9/10 (cloud/cirrus), 11 (snow).
    valid = ~np.isin(scl, [0, 1, 3, 8, 9, 10, 11])
    red = bands["B04"]
    green = bands["B03"]
    blue = bands["B02"]
    nir = bands["B08"]
    ndvi = np.divide(nir - red, nir + red, out=np.zeros_like(red), where=(nir + red) != 0)

    # Percentile stretch for display only; keep reflectance bands out of analytical metrics.
    rgb_stack = np.stack([red, green, blue], axis=-1)
    rgb = np.zeros_like(rgb_stack, dtype=np.uint8)
    for i in range(3):
        band = rgb_stack[..., i]
        vals = band[valid & np.isfinite(band)]
        if vals.size:
            lo, hi = np.percentile(vals, [2, 98])
            if hi > lo:
                scaled = np.clip((band - lo) / (hi - lo), 0, 1)
                rgb[..., i] = (scaled * 255).astype(np.uint8)

    meta = {
        "crs": str(dst_crs),
        "transform": tuple(dst_transform),
        "width": width,
        "height": height,
        "bounds": tuple(src_bounds),
        "date": item.datetime.isoformat() if item.datetime else "unknown",
        "item_id": item.id,
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "valid_pixel_pct": float(valid.mean() * 100),
        "resolution_m": resolution,
    }
    return {"rgb": rgb, "ndvi": ndvi, "valid_mask": valid, "scl": scl, "meta": meta}
