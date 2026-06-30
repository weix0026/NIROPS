# OPTION A IMPLEMENTATION (FULL SCRIPT)
# - Uses REF_RASTER ONLY for CRS + pixel size
# - Output extent comes from LAST perimeter per incident (NOT clipped to REF bounds)
# - Builds a custom 30m grid snapped to a 30m lattice in the REF CRS
# - Filters all inputs to TRUEDATE years 2020 through 2025

import os
import re
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from pyproj import CRS

# =============================================================================
# USER CONFIG
# =============================================================================
REF_RASTER = r"C:\Users\magst\OneDrive\Pictures\Desktop\NIROPS_CNN\InputData\REFERENCERASTER\LC20_SlpD_220.tif"

PERIM_SHP     = r"C:\Users\magst\OneDrive\Pictures\Desktop\NIROPS_CNN\InputData\ALL_NIROPS_2013_2025_Heat_Perimeter.shp"
INTENSE_SHP   = r"C:\Users\magst\OneDrive\Pictures\Desktop\NIROPS_CNN\InputData\ALL_NIROPS_2013_2025_Intense_Heat.shp"
ISOLATED_SHP  = r"C:\Users\magst\OneDrive\Pictures\Desktop\NIROPS_CNN\InputData\ALL_NIROPS_2013_2025_Isolated_Heat.shp"
SCATTERED_SHP = r"C:\Users\magst\OneDrive\Pictures\Desktop\NIROPS_CNN\InputData\ALL_NIROPS_2013_2025_Scattered_Heat.shp"

OUT_DIR = r"C:\Users\magst\OneDrive\Pictures\Desktop\NIROPS_CNN\Rasterized"

INCIDENT_FIELD = "INCIDENTNA"
DATE_FIELD     = "TRUEDATE"
TIME_FIELD     = "TRUETIME"

START_YEAR = 2020
END_YEAR   = 2025

ISOLATED_BUFFER_M = 30.0

# =============================================================================
# DATE/TIME NORMALIZATION
# =============================================================================
def _normalize_date_str(x) -> str:
    if x is None:
        return ""

    s = str(x).strip()

    if s.lower() in ("nan", "none", ""):
        return ""

    # Handle numeric shapefile values like 20251020.0
    if re.match(r"^\d{8}\.0$", s):
        s = s.split(".")[0]

    # Keep only digits if it looks like YYYYMMDD with extra formatting
    digits = re.sub(r"[^\d]", "", s)

    if len(digits) == 8:
        y = digits[0:4]
        m = digits[4:6]
        d = digits[6:8]
        return f"{y}-{m}-{d}"

    s = s.replace("/", "-")

    m = re.match(r"^\d{4}-\d{2}-\d{2}$", s)
    if m:
        return s

    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    return ""

def _normalize_time_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none", ""):
        return ""

    s = s.replace(":", "")
    s = re.sub(r"[^\d]", "", s)

    if len(s) == 3:
        s = "0" + s
    if len(s) == 1:
        s = "0" + s + "00"
    if len(s) == 2:
        s = s + "00"
    if len(s) >= 4:
        s = s[:4]

    return s.zfill(4)

def _dt_key(date_str: str, time_str: str):
    try:
        y, m, d = [int(v) for v in date_str.split("-")]
        hh = int(time_str[:2])
        mm = int(time_str[2:])
        return (y, m, d, hh, mm)
    except Exception:
        return (date_str, time_str)

def _safe_name(s: str) -> str:
    if s is None:
        return "unknown"
    s = str(s).strip()
    if s == "":
        return "unknown"
    s = re.sub(r"[^\w\-]+", "_", s)
    return s[:160]

def _drop_empty(gdf):
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]
    return gdf

def require_fields(gdf, path, fields):
    missing = [f for f in fields if f not in gdf.columns]
    if missing:
        raise ValueError(f"Missing fields in {path}: {missing}\nColumns present: {list(gdf.columns)}")

def add_norm_dt(gdf):
    gdf = gdf.copy()
    gdf["_DATE_N"] = gdf[DATE_FIELD].apply(_normalize_date_str)
    gdf["_TIME_N"] = gdf[TIME_FIELD].apply(_normalize_time_str)
    return gdf

def filter_2020_to_2025(gdf, label):
    if gdf is None or gdf.empty:
        return gdf

    gdf = gdf.copy()

    gdf["_YEAR"] = pd.to_numeric(
        gdf["_DATE_N"].astype(str).str[:4],
        errors="coerce"
    )

    before = len(gdf)

    gdf = gdf[
        (gdf["_YEAR"] >= START_YEAR) &
        (gdf["_YEAR"] <= END_YEAR)
    ].copy()

    after = len(gdf)
    print(f"{label}: kept {after:,} of {before:,} records from {START_YEAR}-{END_YEAR}")

    return gdf

# =============================================================================
# GRID HELPERS
# =============================================================================
def snap_bounds_to_grid(minx, miny, maxx, maxy, res):
    minx_s = math.floor(minx / res) * res
    miny_s = math.floor(miny / res) * res
    maxx_s = math.ceil(maxx / res) * res
    maxy_s = math.ceil(maxy / res) * res
    return minx_s, miny_s, maxx_s, maxy_s

def grid_from_bounds(bounds, res):
    minx, miny, maxx, maxy = bounds
    minx, miny, maxx, maxy = snap_bounds_to_grid(minx, miny, maxx, maxy, res)

    width  = int(round((maxx - minx) / res))
    height = int(round((maxy - miny) / res))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid grid size from bounds={bounds} at res={res}: height={height}, width={width}")

    transform = from_origin(minx, maxy, res, res)
    return height, width, transform

# =============================================================================
# IO / GEOMETRY HELPERS
# =============================================================================
def load_and_fix_shp(path, target_crs):
    if not path or not os.path.exists(path):
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)

    gdf = gpd.read_file(path)

    if gdf.empty:
        if gdf.crs is not None:
            return gdf.to_crs(target_crs)
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)

    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS; set it before reprojecting.")

    gdf = gdf.to_crs(target_crs)
    gdf = _drop_empty(gdf)

    if not gdf.empty:
        poly_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        if poly_mask.any():
            gdf = gdf.copy()
            gdf.loc[poly_mask, "geometry"] = gdf.loc[poly_mask, "geometry"].buffer(0)
            gdf = _drop_empty(gdf)

    return gdf

def buffer_isolated_points_to_polys(gdf, buffer_m, ref_crs):
    if gdf is None or gdf.empty:
        return gdf

    gdf = _drop_empty(gdf)
    if gdf.empty:
        return gdf

    point_mask = gdf.geometry.geom_type.isin(["Point", "MultiPoint"])
    if not point_mask.any():
        return gdf

    ref = CRS.from_user_input(ref_crs)

    if ref.is_geographic:
        gdf_wgs = gdf.to_crs(4326)
        c = gdf_wgs.unary_union.centroid
        lon, lat = float(c.x), float(c.y)
        zone = int((lon + 180) // 6) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone

        gdf_utm = gdf.to_crs(epsg)
        gdf_utm = gdf_utm.copy()
        gdf_utm.loc[point_mask.values, "geometry"] = gdf_utm.loc[point_mask.values, "geometry"].buffer(buffer_m)
        gdf_utm = _drop_empty(gdf_utm)

        return gdf_utm.to_crs(ref_crs)

    gdf = gdf.copy()
    gdf.loc[point_mask, "geometry"] = gdf.loc[point_mask, "geometry"].buffer(buffer_m)
    gdf = _drop_empty(gdf)

    return gdf

def rasterize_layer(gdf, value, out_shape, transform, dtype="int16"):
    if gdf is None or gdf.empty:
        return np.zeros(out_shape, dtype=dtype)

    shapes = [
        (geom, value)
        for geom in gdf.geometry
        if geom is not None and not geom.is_empty
    ]

    if not shapes:
        return np.zeros(out_shape, dtype=dtype)

    return rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=dtype,
        all_touched=False,
    )

# =============================================================================
# LOAD REF RASTER METADATA
# =============================================================================
with rasterio.open(REF_RASTER) as ref:
    ref_profile   = ref.profile.copy()
    ref_transform = ref.transform
    ref_crs       = ref.crs

RES_M = float(abs(ref_transform.a))

print("REF_RASTER CRS:", ref_crs)
print("REF_RASTER pixel size:", RES_M)

# =============================================================================
# LOAD MERGED SHAPEFILES
# =============================================================================
perim_all     = load_and_fix_shp(PERIM_SHP, ref_crs)
intense_all   = load_and_fix_shp(INTENSE_SHP, ref_crs)
isolated_all  = load_and_fix_shp(ISOLATED_SHP, ref_crs)
scattered_all = load_and_fix_shp(SCATTERED_SHP, ref_crs)

required = [INCIDENT_FIELD, DATE_FIELD, TIME_FIELD]

require_fields(perim_all, PERIM_SHP, required)
require_fields(intense_all, INTENSE_SHP, required)
require_fields(isolated_all, ISOLATED_SHP, required)
require_fields(scattered_all, SCATTERED_SHP, required)

perim_all     = add_norm_dt(perim_all)
intense_all   = add_norm_dt(intense_all)
isolated_all  = add_norm_dt(isolated_all)
scattered_all = add_norm_dt(scattered_all)

# =============================================================================
# FILTER ALL INPUTS TO TRUEDATE YEARS 2020-2025
# =============================================================================
perim_all     = filter_2020_to_2025(perim_all, "Perimeters")
intense_all   = filter_2020_to_2025(intense_all, "Intense heat")
isolated_all  = filter_2020_to_2025(isolated_all, "Isolated heat")
scattered_all = filter_2020_to_2025(scattered_all, "Scattered heat")

incidents = sorted(
    perim_all[INCIDENT_FIELD]
    .dropna()
    .unique()
    .tolist()
)

if not incidents:
    raise RuntimeError(f"No incidents found in perimeter shapefile from {START_YEAR}-{END_YEAR} using {DATE_FIELD}.")

print(f"Found {len(incidents)} incidents from {START_YEAR}-{END_YEAR}.")

# =============================================================================
# PROCESS EACH INCIDENT
# =============================================================================
os.makedirs(OUT_DIR, exist_ok=True)

for incident in incidents:
    incident_name = _safe_name(incident)

    print("\n" + "=" * 80)
    print(f"INCIDENT: {incident_name}")

    perim_i     = perim_all[perim_all[INCIDENT_FIELD] == incident].copy()
    intense_i   = intense_all[intense_all[INCIDENT_FIELD] == incident].copy()
    isolated_i  = isolated_all[isolated_all[INCIDENT_FIELD] == incident].copy()
    scattered_i = scattered_all[scattered_all[INCIDENT_FIELD] == incident].copy()

    if perim_i.empty:
        print("  Skipping: no perimeters for this incident.")
        continue

    ts = perim_i[["_DATE_N", "_TIME_N"]].drop_duplicates()
    ts = ts[(ts["_DATE_N"] != "") & (ts["_TIME_N"] != "")]

    if ts.empty:
        print("  Skipping: no valid DATE/TIME in perimeters after normalization.")
        continue

    ts_list_sorted = sorted(ts.values.tolist(), key=lambda x: _dt_key(x[0], x[1]))

    print(f"  Timestamps from perimeter data: {len(ts_list_sorted)}")
    print(f"  First timestamp: {ts_list_sorted[0][0]} {ts_list_sorted[0][1]}")
    print(f"  Last timestamp:  {ts_list_sorted[-1][0]} {ts_list_sorted[-1][1]}")

    last_date, last_time = ts_list_sorted[-1]

    last_perim = perim_i[
        (perim_i["_DATE_N"] == last_date) &
        (perim_i["_TIME_N"] == last_time)
    ].copy()

    last_perim = _drop_empty(last_perim)

    if last_perim.empty:
        print("  Skipping: last timestamp perimeter is empty.")
        continue

    bounds = last_perim.total_bounds

    try:
        out_h, out_w, out_transform = grid_from_bounds(bounds, RES_M)
    except Exception as e:
        print("  Skipping: could not build grid from bounds:", e)
        continue

    print(f"  Output grid from last perimeter: {last_date} {last_time}")
    print(f"  Bounds: minx={bounds[0]:.3f}, miny={bounds[1]:.3f}, maxx={bounds[2]:.3f}, maxy={bounds[3]:.3f}")
    print(f"  Grid: height={out_h}, width={out_w}, res={RES_M}")

    out_profile = ref_profile.copy()
    out_profile.update(
        driver="GTiff",
        height=out_h,
        width=out_w,
        transform=out_transform,
        crs=ref_crs,
        dtype="int16",
        count=1,
        nodata=0,
        compress="deflate",
    )

    incident_out_dir = os.path.join(OUT_DIR, incident_name)
    os.makedirs(incident_out_dir, exist_ok=True)

    cumulative_in_perim = np.zeros((out_h, out_w), dtype=bool)

    for date_n, time_n in ts_list_sorted:
        label = f"{date_n.replace('-', '')}_{time_n}"

        print(f"  Processing: {label}")

        perim_gdf = _drop_empty(
            perim_i[
                (perim_i["_DATE_N"] == date_n) &
                (perim_i["_TIME_N"] == time_n)
            ].copy()
        )

        intense_gdf = _drop_empty(
            intense_i[
                (intense_i["_DATE_N"] == date_n) &
                (intense_i["_TIME_N"] == time_n)
            ].copy()
        )

        scattered_gdf = _drop_empty(
            scattered_i[
                (scattered_i["_DATE_N"] == date_n) &
                (scattered_i["_TIME_N"] == time_n)
            ].copy()
        )

        isolated_gdf = _drop_empty(
            isolated_i[
                (isolated_i["_DATE_N"] == date_n) &
                (isolated_i["_TIME_N"] == time_n)
            ].copy()
        )

        isolated_gdf = buffer_isolated_points_to_polys(
            isolated_gdf,
            ISOLATED_BUFFER_M,
            ref_crs
        )

        perim_mask = rasterize_layer(
            perim_gdf,
            1,
            (out_h, out_w),
            out_transform,
            dtype="int16"
        ) > 0

        intense_mask = rasterize_layer(
            intense_gdf,
            1,
            (out_h, out_w),
            out_transform,
            dtype="int16"
        ) > 0

        scattered_mask = rasterize_layer(
            scattered_gdf,
            1,
            (out_h, out_w),
            out_transform,
            dtype="int16"
        ) > 0

        isolated_mask = rasterize_layer(
            isolated_gdf,
            1,
            (out_h, out_w),
            out_transform,
            dtype="int16"
        ) > 0

        class2_mask = scattered_mask | isolated_mask

        cumulative_in_perim |= perim_mask

        out_arr = np.zeros((out_h, out_w), dtype="int16")

        interior_mask = cumulative_in_perim & ~intense_mask & ~class2_mask

        out_arr[interior_mask] = 3
        out_arr[class2_mask]   = 2
        out_arr[intense_mask]  = 1

        out_path = os.path.join(
            incident_out_dir,
            f"{incident_name}_{label}_PERIM_RASTERIZED.tif"
        )

        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(out_arr, 1)

        u = np.unique(out_arr)

        print(f"    Wrote: {out_path}")
        print(f"    Unique values: {u}")

print(f"\nDone. Built cumulative NIROPS rasters per INCIDENTNA for TRUEDATE years {START_YEAR}-{END_YEAR}.")
