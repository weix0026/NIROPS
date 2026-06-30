from pathlib import Path
from collections import defaultdict
import re
import warnings
import pandas as pd
import geopandas as gpd
import fiona

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# INPUT / OUTPUT
# ============================================================

input_gpkg = Path(
    r"C:\Users\magst\OneDrive\Pictures\Desktop\NIROPSCLEANING\GPKG_YEAR\NIROPS_2025.gpkg"
)

out_dir = Path(
    r"C:\Users\magst\OneDrive\Pictures\Desktop\NIROPSCLEANING\GPKG_YEAR\CLEANED"
)
out_dir.mkdir(parents=True, exist_ok=True)

out_cleaned_gpkg = out_dir / "NIROPS_2025_CLEANED.gpkg"
out_removed_gpkg = out_dir / "NIROPS_2025_REMOVED.gpkg"

layers = ["scattered_heat", "heat_perimeter", "intense_heat", "isolated_heat"]

date_col = "TRUEDATE"
time_col = "TRUETIME"
incident_col = "INCIDENTNA"
year_value = 2025

project_crs = "EPSG:5070"

# ============================================================
# QAQC PARAMETERS
# ============================================================

# True = if one record fails, remove the whole incident/date from all layers.
# False = remove only the exact incident/date/time from all layers.
delete_entire_calendar_date = True

# Heat perimeter is allowed to shrink only a tiny amount.
area_dip_tolerance_rel = 0.02
area_dip_tolerance_abs_km2 = 0.10

# Require scattered/intense/isolated heat to have a matching heat perimeter.
require_perimeter_for_heat = True

# Require each perimeter observation to have at least one non-perimeter heat layer.
require_heat_with_perimeter = True

# Final sequence check:
# After all other QAQC removals, each incident must still have at least
# 3 sequential heat_perimeter observations where each gap is <= 36 hours.
min_sequential_observations = 3
max_gap_hours = 36.0

# Temporal island cleanup:
# If observations are split by more than this many days, keep the main cluster
# and remove isolated old/new temporal islands.
temporal_island_gap_days = 30.0
remove_non_main_temporal_islands = True

# If True, when an incident has multiple temporal clusters separated by >30 days,
# keep the largest cluster. If tied, keep the latest cluster.
# This removes lone old points like 2025-06-01 before a September sequence.
keep_largest_temporal_cluster = True

# Local high spike cleanup:
# Removes one-observation high spikes such as low -> high -> low.
local_spike_rel_threshold = 0.02
local_spike_abs_threshold_km2 = 0.10

# Major low-collapse cleanup:
# Removes only large collapses such as high -> zero/very low -> high.
# This avoids removing normal middle points between two high spikes.
low_dip_max_fraction_of_context = 0.50
low_dip_near_zero_km2 = 0.01
low_dip_context_n = 2

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text_date(value):
    if pd.isna(value):
        return ""
    s = str(value).strip()
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"\D", "", s)
    return s.zfill(8) if s else ""


def clean_text_time(value):
    if pd.isna(value):
        return ""
    s = str(value).strip()
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"\D", "", s)
    return s.zfill(4) if s else ""


def parse_datetime_fields(gdf):
    gdf = gdf.copy()

    gdf["_incident_str"] = gdf[incident_col].astype(str).str.strip()
    gdf["_date_str"] = gdf[date_col].apply(clean_text_date)
    gdf["_time_str"] = gdf[time_col].apply(clean_text_time)

    parsed_date = pd.to_datetime(gdf["_date_str"], format="%Y%m%d", errors="coerce")

    date_ok = (
        gdf["_date_str"].str.match(r"^\d{8}$", na=False)
        & parsed_date.notna()
        & (parsed_date.dt.year == year_value)
    )

    hour = pd.to_numeric(gdf["_time_str"].str.slice(0, 2), errors="coerce")
    minute = pd.to_numeric(gdf["_time_str"].str.slice(2, 4), errors="coerce")

    time_ok = (
        gdf["_time_str"].str.match(r"^\d{4}$", na=False)
        & hour.between(0, 23)
        & minute.between(0, 59)
    )

    gdf["_datetime"] = pd.to_datetime(
        gdf["_date_str"] + " " + gdf["_time_str"],
        format="%Y%m%d %H%M",
        errors="coerce",
    )

    gdf["_datetime_str"] = gdf["_datetime"].dt.strftime("%Y-%m-%d %H:%M")

    gdf["_obs_key"] = (
        gdf["_incident_str"]
        + "|"
        + gdf["_date_str"]
        + "|"
        + gdf["_time_str"]
    )

    if delete_entire_calendar_date:
        gdf["_remove_key"] = (
            gdf["_incident_str"]
            + "|"
            + gdf["_date_str"]
        )
    else:
        gdf["_remove_key"] = gdf["_obs_key"]

    gdf["_date_time_ok"] = date_ok & time_ok & gdf["_datetime"].notna()

    return gdf


def make_valid_safe(geom):
    if geom is None:
        return geom

    try:
        if geom.is_empty:
            return geom
    except Exception:
        return geom

    try:
        if geom.is_valid:
            return geom
    except Exception:
        return geom

    try:
        from shapely import make_valid
        fixed = make_valid(geom)
    except Exception:
        try:
            fixed = geom.buffer(0)
        except Exception:
            fixed = geom

    return fixed


def add_bad_day(bad_day_reasons, bad_day_details, row, reason, detail):
    remove_key = row["_remove_key"]
    obs_key = row["_obs_key"]

    bad_day_reasons[remove_key].add(reason)

    if detail:
        bad_day_details[remove_key].append(f"{reason}: obs={obs_key}; {detail}")
    else:
        bad_day_details[remove_key].append(f"{reason}: obs={obs_key}")


def add_bad_incident(bad_incident_reasons, bad_incident_details, incident, reason, detail):
    incident = str(incident).strip()
    bad_incident_reasons[incident].add(reason)

    if detail:
        bad_incident_details[incident].append(f"{reason}: {detail}")
    else:
        bad_incident_details[incident].append(reason)


def valid_layer(data, layer, bad_day_reasons):
    bad_keys = set(bad_day_reasons.keys())

    gdf = data[layer].copy()

    mask = (
        gdf["_date_time_ok"]
        & gdf["_geom_ok"]
        & ~gdf["_remove_key"].isin(bad_keys)
    )

    return gdf.loc[mask].copy()


def has_required_sequence(datetimes, min_n=3, max_gap_h=36.0):
    dts = pd.Series(datetimes).dropna().drop_duplicates().sort_values().tolist()

    if len(dts) < min_n:
        return False, len(dts), None

    run_len = 1
    best_run = 1

    for i in range(1, len(dts)):
        gap_h = (dts[i] - dts[i - 1]).total_seconds() / 3600.0

        if gap_h <= max_gap_h:
            run_len += 1
        else:
            run_len = 1

        best_run = max(best_run, run_len)

        if run_len >= min_n:
            return True, len(dts), best_run

    return False, len(dts), best_run


def clean_for_output(gdf, original_crs):
    out = gdf.copy()

    out[date_col] = out["_date_str"]
    out[time_col] = out["_time_str"]
    out["year"] = year_value

    drop_cols = [
        "_orig_index",
        "_source_layer",
        "_incident_str",
        "_date_str",
        "_time_str",
        "_datetime",
        "_datetime_str",
        "_obs_key",
        "_remove_key",
        "_date_time_ok",
        "_geom_ok",
        "_area_km2",
    ]

    out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")
    return out.to_crs(original_crs)


def removed_for_output(gdf, original_crs):
    out = gdf.copy()

    out = out.drop(columns=["_datetime"], errors="ignore")

    rename = {
        "_orig_index": "orig_index",
        "_source_layer": "src_layer",
        "_incident_str": "inc_str",
        "_date_str": "date_str",
        "_time_str": "time_str",
        "_datetime_str": "dt_str",
        "_obs_key": "obs_key",
        "_remove_key": "remove_key",
        "_date_time_ok": "dt_ok",
        "_geom_ok": "geom_ok",
        "_area_km2": "area_km2",
    }

    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    return out.to_crs(original_crs)


def allowed_area_difference(reference_area_km2, rel_threshold, abs_threshold_km2):
    return max(
        float(abs_threshold_km2),
        float(rel_threshold) * max(float(reference_area_km2), 0.0),
    )


def median_or_none(values):
    vals = pd.Series(values).dropna()
    if len(vals) == 0:
        return None
    return float(vals.median())


def build_perimeter_observation_area_table(hp_valid):
    rows = []

    group_cols = [
        "_incident_str",
        "_obs_key",
        "_remove_key",
        "_datetime",
        "_datetime_str",
    ]

    for vals, sub in hp_valid.groupby(group_cols, dropna=False):
        incident, obs_key, remove_key, dt, dt_str = vals

        try:
            dissolved_geom = sub.geometry.unary_union
            area_km2 = dissolved_geom.area / 1_000_000.0
        except Exception:
            area_km2 = sub.geometry.area.sum() / 1_000_000.0

        rows.append(
            {
                "_incident_str": str(incident).strip(),
                "_obs_key": obs_key,
                "_remove_key": remove_key,
                "_datetime": dt,
                "_datetime_str": dt_str,
                "_area_km2": float(area_km2),
                "_n_perimeter_parts": int(len(sub)),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "_incident_str",
            "_obs_key",
            "_remove_key",
            "_datetime",
            "_datetime_str",
            "_area_km2",
            "_n_perimeter_parts",
        ],
    )


def mark_bad_obs_key(obs_key, reason, detail, hp_representative_rows):
    if obs_key not in hp_representative_rows.index:
        return

    row = hp_representative_rows.loc[obs_key]

    add_bad_day(
        bad_day_reasons,
        bad_day_details,
        row,
        reason,
        detail,
    )


# ============================================================
# LOAD ALL LAYERS
# ============================================================

available_layers = fiona.listlayers(input_gpkg)

missing_layers = [layer for layer in layers if layer not in available_layers]
if missing_layers:
    raise ValueError(
        f"Missing expected layers: {missing_layers}. Available layers: {available_layers}"
    )

first = gpd.read_file(input_gpkg, layer=layers[0], rows=1)
original_crs = first.crs

if original_crs is None:
    raise ValueError("Input GeoPackage has no CRS. Set the CRS before running QAQC.")

data = {}

bad_day_reasons = defaultdict(set)
bad_day_details = defaultdict(list)

bad_incident_reasons = defaultdict(set)
bad_incident_details = defaultdict(list)

for layer in layers:
    print(f"\nReading {layer}")

    gdf = gpd.read_file(input_gpkg, layer=layer)

    if incident_col not in gdf.columns:
        raise ValueError(f"{incident_col} not found in {layer}")

    if date_col not in gdf.columns or time_col not in gdf.columns:
        raise ValueError(f"{date_col} or {time_col} not found in {layer}")

    gdf["_orig_index"] = range(len(gdf))
    gdf["_source_layer"] = layer

    gdf = parse_datetime_fields(gdf)
    gdf = gdf.to_crs(project_crs)

    gdf["geometry"] = gdf.geometry.apply(make_valid_safe)

    gdf["_geom_ok"] = ~(gdf.geometry.isna() | gdf.geometry.is_empty)

    for _, row in gdf.loc[~gdf["_date_time_ok"]].iterrows():
        add_bad_day(
            bad_day_reasons,
            bad_day_details,
            row,
            "BAD_DATE_TIME",
            "TRUEDATE must be YYYYMMDD and TRUETIME must be valid HHMM from 0000 to 2359",
        )

    for _, row in gdf.loc[~gdf["_geom_ok"]].iterrows():
        add_bad_day(
            bad_day_reasons,
            bad_day_details,
            row,
            "BAD_GEOMETRY",
            "geometry was null, empty, or could not be repaired",
        )

    data[layer] = gdf

    print(f"  total rows: {len(gdf)}")
    print(f"  bad date/time rows: {(~gdf['_date_time_ok']).sum()}")
    print(f"  bad geometry rows: {(~gdf['_geom_ok']).sum()}")


# ============================================================
# QAQC 1: NON-PERIMETER HEAT MUST HAVE MATCHING HEAT PERIMETER
# ============================================================

if require_perimeter_for_heat:
    hp_valid = valid_layer(data, "heat_perimeter", bad_day_reasons)
    perimeter_obs_keys = set(hp_valid["_obs_key"].unique().tolist())

    for layer in ["scattered_heat", "intense_heat", "isolated_heat"]:
        gdf = valid_layer(data, layer, bad_day_reasons)

        no_perimeter = ~gdf["_obs_key"].isin(perimeter_obs_keys)

        for _, row in gdf.loc[no_perimeter].iterrows():
            add_bad_day(
                bad_day_reasons,
                bad_day_details,
                row,
                "NON_PERIMETER_HEAT_WITHOUT_MATCHING_PERIMETER",
                f"{layer} row had no matching heat_perimeter row for the same incident/date/time",
            )


# ============================================================
# QAQC 2: HEAT PERIMETER MUST HAVE SOME NON-PERIMETER HEAT
# ============================================================

if require_heat_with_perimeter:
    hp_valid = valid_layer(data, "heat_perimeter", bad_day_reasons)

    non_perimeter_obs_keys = set()

    for layer in ["scattered_heat", "intense_heat", "isolated_heat"]:
        gdf = valid_layer(data, layer, bad_day_reasons)
        non_perimeter_obs_keys.update(gdf["_obs_key"].unique().tolist())

    perimeter_only = hp_valid.loc[
        ~hp_valid["_obs_key"].isin(non_perimeter_obs_keys)
    ].copy()

    for _, row in perimeter_only.iterrows():
        add_bad_day(
            bad_day_reasons,
            bad_day_details,
            row,
            "PERIMETER_WITH_NO_NON_PERIMETER_HEAT",
            "heat_perimeter existed but no scattered_heat, intense_heat, or isolated_heat existed for the same incident/date/time after QAQC",
        )


# ============================================================
# QAQC 3: OBSERVATION-LEVEL HEAT PERIMETER AREA / TIME CLEANUP
#
# Removes:
#   1. Local high spikes: low -> high -> low
#   2. Major local low collapses: high -> zero/very low -> high
#   3. Isolated temporal islands separated by more than 30 days
#
# This removes bad observations/dates first.
# It does not automatically remove the whole incident unless the
# final sequence check fails after these observation-level removals.
# ============================================================

hp_valid = valid_layer(data, "heat_perimeter", bad_day_reasons)

hp_obs = build_perimeter_observation_area_table(hp_valid)

hp_representative_rows = (
    hp_valid.sort_values("_datetime")
    .drop_duplicates("_obs_key")
    .set_index("_obs_key", drop=False)
)

bad_area_obs_keys = set()
bad_temporal_obs_keys = set()


# ------------------------------------------------------------
# 3A: Remove temporal islands separated by >30 days.
# ------------------------------------------------------------

if remove_non_main_temporal_islands and len(hp_obs) > 0:
    for incident, grp in hp_obs.sort_values("_datetime").groupby("_incident_str"):
        grp = grp.sort_values("_datetime").reset_index(drop=True)

        if len(grp) <= 1:
            continue

        cluster_ids = []
        cluster_id = 0

        for i in range(len(grp)):
            if i == 0:
                cluster_ids.append(cluster_id)
                continue

            gap_days = (
                grp.loc[i, "_datetime"] - grp.loc[i - 1, "_datetime"]
            ).total_seconds() / 86400.0

            if gap_days > temporal_island_gap_days:
                cluster_id += 1

            cluster_ids.append(cluster_id)

        grp["_temporal_cluster_id"] = cluster_ids

        cluster_summary = (
            grp.groupby("_temporal_cluster_id")
            .agg(
                n_obs=("_obs_key", "count"),
                first_dt=("_datetime", "min"),
                last_dt=("_datetime", "max"),
            )
            .reset_index()
        )

        if len(cluster_summary) <= 1:
            continue

        if keep_largest_temporal_cluster:
            cluster_summary = cluster_summary.sort_values(
                ["n_obs", "last_dt"],
                ascending=[False, False],
            )
            keep_cluster_id = cluster_summary.iloc[0]["_temporal_cluster_id"]
        else:
            cluster_summary = cluster_summary.sort_values(
                "last_dt",
                ascending=False,
            )
            keep_cluster_id = cluster_summary.iloc[0]["_temporal_cluster_id"]

        kept = grp.loc[
            grp["_temporal_cluster_id"] == keep_cluster_id
        ].copy()

        kept_first = kept["_datetime"].min()
        kept_last = kept["_datetime"].max()

        remove_grp = grp.loc[
            grp["_temporal_cluster_id"] != keep_cluster_id
        ].copy()

        for _, obs in remove_grp.iterrows():
            bad_temporal_obs_keys.add(obs["_obs_key"])

            detail = (
                f"heat_perimeter observation was in a temporal island separated by "
                f"more than {temporal_island_gap_days:.1f} days from the main observation cluster; "
                f"removed observation at {obs['_datetime_str']} with area {obs['_area_km2']:.3f} km2; "
                f"kept main cluster from {kept_first} to {kept_last}"
            )

            mark_bad_obs_key(
                obs["_obs_key"],
                "TEMPORAL_ISLAND_OBSERVATION",
                detail,
                hp_representative_rows,
            )


# ------------------------------------------------------------
# 3B-1: Remove local HIGH spikes only.
#
# This catches:
#   normal -> artificial high -> normal
#
# This does NOT remove low middle points.
# ------------------------------------------------------------

if len(hp_obs) > 0:
    for incident, grp in hp_obs.sort_values("_datetime").groupby("_incident_str"):
        grp = grp.sort_values("_datetime").reset_index(drop=True)

        changed = True

        while changed:
            changed = False

            kept = grp.loc[
                ~grp["_obs_key"].isin(bad_area_obs_keys)
                & ~grp["_obs_key"].isin(bad_temporal_obs_keys)
            ].copy()

            kept = kept.sort_values("_datetime").reset_index(drop=True)

            if len(kept) < 3:
                break

            newly_bad = []

            for i in range(1, len(kept) - 1):
                prev_row = kept.iloc[i - 1]
                cur_row = kept.iloc[i]
                next_row = kept.iloc[i + 1]

                prev_area = float(prev_row["_area_km2"])
                cur_area = float(cur_row["_area_km2"])
                next_area = float(next_row["_area_km2"])

                high_spike_tol_prev = allowed_area_difference(
                    max(prev_area, cur_area),
                    local_spike_rel_threshold,
                    local_spike_abs_threshold_km2,
                )

                high_spike_tol_next = allowed_area_difference(
                    max(next_area, cur_area),
                    local_spike_rel_threshold,
                    local_spike_abs_threshold_km2,
                )

                is_local_high_spike = (
                    cur_area > prev_area + high_spike_tol_prev
                    and cur_area > next_area + high_spike_tol_next
                )

                if is_local_high_spike:
                    newly_bad.append(
                        (
                            cur_row["_obs_key"],
                            "PERIMETER_LOCAL_AREA_HIGH_SPIKE",
                            (
                                f"heat_perimeter observation was a local high area spike; "
                                f"previous={prev_area:.3f} km2 at {prev_row['_datetime_str']}, "
                                f"current={cur_area:.3f} km2 at {cur_row['_datetime_str']}, "
                                f"next={next_area:.3f} km2 at {next_row['_datetime_str']}; "
                                f"removed high spike observation"
                            ),
                        )
                    )

            if newly_bad:
                for obs_key, reason, detail in newly_bad:
                    if obs_key not in bad_area_obs_keys:
                        bad_area_obs_keys.add(obs_key)
                        mark_bad_obs_key(
                            obs_key,
                            reason,
                            detail,
                            hp_representative_rows,
                        )

                changed = True


# ------------------------------------------------------------
# 3B-2: Remove MAJOR local LOW collapses only.
#
# This catches:
#   high -> zero/very low -> high
#
# It avoids removing normal in-between points like MoonComplex,
# because those are not major collapses relative to the surrounding context.
# ------------------------------------------------------------

if len(hp_obs) > 0:
    for incident, grp in hp_obs.sort_values("_datetime").groupby("_incident_str"):
        grp = grp.sort_values("_datetime").reset_index(drop=True)

        changed = True

        while changed:
            changed = False

            kept = grp.loc[
                ~grp["_obs_key"].isin(bad_area_obs_keys)
                & ~grp["_obs_key"].isin(bad_temporal_obs_keys)
            ].copy()

            kept = kept.sort_values("_datetime").reset_index(drop=True)

            if len(kept) < 3:
                break

            newly_bad = []

            for i in range(1, len(kept) - 1):
                cur_row = kept.iloc[i]
                cur_area = float(cur_row["_area_km2"])

                before_rows = kept.iloc[max(0, i - low_dip_context_n):i]
                after_rows = kept.iloc[i + 1:min(len(kept), i + 1 + low_dip_context_n)]

                if len(before_rows) == 0 or len(after_rows) == 0:
                    continue

                before_context = median_or_none(before_rows["_area_km2"].tolist())
                after_context = median_or_none(after_rows["_area_km2"].tolist())

                if before_context is None or after_context is None:
                    continue

                if before_context <= area_dip_tolerance_abs_km2:
                    continue

                if after_context <= area_dip_tolerance_abs_km2:
                    continue

                is_near_zero_collapse = (
                    cur_area <= low_dip_near_zero_km2
                    and before_context > area_dip_tolerance_abs_km2
                    and after_context > area_dip_tolerance_abs_km2
                )

                is_major_low_collapse = (
                    cur_area <= low_dip_max_fraction_of_context * before_context
                    and cur_area <= low_dip_max_fraction_of_context * after_context
                    and before_context > area_dip_tolerance_abs_km2
                    and after_context > area_dip_tolerance_abs_km2
                )

                if is_near_zero_collapse or is_major_low_collapse:
                    newly_bad.append(
                        (
                            cur_row["_obs_key"],
                            "PERIMETER_MAJOR_LOCAL_AREA_LOW_COLLAPSE",
                            (
                                f"heat_perimeter observation was a major local low-area collapse; "
                                f"before_context_median={before_context:.3f} km2, "
                                f"current={cur_area:.3f} km2 at {cur_row['_datetime_str']}, "
                                f"after_context_median={after_context:.3f} km2; "
                                f"removed only because current area was <= "
                                f"{low_dip_max_fraction_of_context:.2f} of both surrounding contexts "
                                f"or near zero"
                            ),
                        )
                    )

            if newly_bad:
                for obs_key, reason, detail in newly_bad:
                    if obs_key not in bad_area_obs_keys:
                        bad_area_obs_keys.add(obs_key)
                        mark_bad_obs_key(
                            obs_key,
                            reason,
                            detail,
                            hp_representative_rows,
                        )

                changed = True


# ------------------------------------------------------------
# 3C: Remaining monotonic decrease check after removing local
# spikes, major collapses, and temporal islands.
#
# This catches remaining perimeter decreases that are not just
# one-day artifacts.
# ------------------------------------------------------------

hp_obs_after_area_cleanup = hp_obs.loc[
    ~hp_obs["_obs_key"].isin(bad_area_obs_keys)
    & ~hp_obs["_obs_key"].isin(bad_temporal_obs_keys)
].copy()

if len(hp_obs_after_area_cleanup) > 0:
    for incident, grp in hp_obs_after_area_cleanup.sort_values("_datetime").groupby("_incident_str"):
        previous_accepted_area = None
        previous_accepted_dt = None

        for _, obs in grp.sort_values("_datetime").iterrows():
            if obs["_remove_key"] in bad_day_reasons:
                continue

            area = float(obs["_area_km2"])

            if previous_accepted_area is None:
                previous_accepted_area = area
                previous_accepted_dt = obs["_datetime"]
                continue

            allowed_drop = allowed_area_difference(
                previous_accepted_area,
                area_dip_tolerance_rel,
                area_dip_tolerance_abs_km2,
            )

            minimum_allowed_area = previous_accepted_area - allowed_drop

            if area < minimum_allowed_area:
                detail = (
                    f"heat_perimeter observation area decreased from "
                    f"{previous_accepted_area:.3f} km2 at {previous_accepted_dt} "
                    f"to {area:.3f} km2 at {obs['_datetime']}; "
                    f"allowed drop was {allowed_drop:.3f} km2"
                )

                mark_bad_obs_key(
                    obs["_obs_key"],
                    "PERIMETER_AREA_DECREASE_AFTER_ARTIFACT_REMOVAL",
                    detail,
                    hp_representative_rows,
                )
            else:
                previous_accepted_area = area
                previous_accepted_dt = obs["_datetime"]


print("\nQAQC 3 observation-level perimeter area/time cleanup")
print(f"  temporal island observations removed: {len(bad_temporal_obs_keys)}")
print(f"  high spike / major low collapse observations removed: {len(bad_area_obs_keys)}")


# ============================================================
# QAQC 4: AFTER ALL DAY-LEVEL REMOVALS, INCIDENT MUST HAVE
# AT LEAST 3 SEQUENTIAL OBSERVATIONS WITH <= 36 HOURS BETWEEN THEM
# ============================================================

hp_after_checks = valid_layer(data, "heat_perimeter", bad_day_reasons)

hp_after_checks_obs = (
    hp_after_checks.sort_values("_datetime")
    .drop_duplicates("_obs_key")
    .copy()
)

for incident, grp in hp_after_checks_obs.groupby("_incident_str"):
    ok, n_obs, best_run = has_required_sequence(
        grp["_datetime"],
        min_n=min_sequential_observations,
        max_gap_h=max_gap_hours,
    )

    if not ok:
        add_bad_incident(
            bad_incident_reasons,
            bad_incident_details,
            str(incident),
            "INSUFFICIENT_SEQUENTIAL_OBSERVATIONS",
            (
                f"incident had {n_obs} remaining heat_perimeter observations after QAQC; "
                f"best sequential run was {best_run}; required at least "
                f"{min_sequential_observations} observations with gaps <= {max_gap_hours} hours"
            ),
        )

all_incidents = set()

for layer in layers:
    all_incidents.update(data[layer]["_incident_str"].astype(str).unique().tolist())

incidents_with_hp_after_checks = set(
    hp_after_checks_obs["_incident_str"].astype(str).unique().tolist()
)

for incident in sorted(all_incidents - incidents_with_hp_after_checks):
    add_bad_incident(
        bad_incident_reasons,
        bad_incident_details,
        str(incident),
        "NO_REMAINING_HEAT_PERIMETER_AFTER_QAQC",
        "incident had no valid heat_perimeter observations remaining after QAQC",
    )


# ============================================================
# FINAL CLEAN / REMOVED SPLIT
# ============================================================

bad_remove_keys = set(bad_day_reasons.keys())
bad_incidents = set(bad_incident_reasons.keys())

cleaned = {}
removed = {}

for layer in layers:
    gdf = data[layer].copy()

    removed_mask = (
        gdf["_remove_key"].isin(bad_remove_keys)
        | gdf["_incident_str"].astype(str).isin(bad_incidents)
        | ~gdf["_date_time_ok"]
        | ~gdf["_geom_ok"]
    )

    cleaned[layer] = gdf.loc[~removed_mask].copy()
    removed[layer] = gdf.loc[removed_mask].copy()

    if len(removed[layer]) > 0:
        removed[layer]["rm_day_reason"] = removed[layer]["_remove_key"].apply(
            lambda k: "; ".join(sorted(bad_day_reasons.get(k, set())))
        )

        removed[layer]["rm_day_detail"] = removed[layer]["_remove_key"].apply(
            lambda k: " || ".join(bad_day_details.get(k, []))
        )

        removed[layer]["rm_inc_reason"] = removed[layer]["_incident_str"].astype(str).apply(
            lambda k: "; ".join(sorted(bad_incident_reasons.get(k, set())))
        )

        removed[layer]["rm_inc_detail"] = removed[layer]["_incident_str"].astype(str).apply(
            lambda k: " || ".join(bad_incident_details.get(k, []))
        )

        removed[layer]["delete_unit"] = removed[layer].apply(
            lambda row: (
                "incident"
                if str(row["_incident_str"]) in bad_incidents
                else ("incident_date" if delete_entire_calendar_date else "incident_datetime")
            ),
            axis=1,
        )

        removed[layer]["rm_reason"] = removed[layer].apply(
            lambda row: "; ".join(
                [
                    x for x in [
                        row.get("rm_day_reason", ""),
                        row.get("rm_inc_reason", ""),
                    ]
                    if isinstance(x, str) and x.strip()
                ]
            ),
            axis=1,
        )

        removed[layer]["rm_detail"] = removed[layer].apply(
            lambda row: " || ".join(
                [
                    x for x in [
                        row.get("rm_day_detail", ""),
                        row.get("rm_inc_detail", ""),
                    ]
                    if isinstance(x, str) and x.strip()
                ]
            ),
            axis=1,
        )

    print(f"\n{layer}")
    print(f"  cleaned rows: {len(cleaned[layer])}")
    print(f"  removed rows: {len(removed[layer])}")


# ============================================================
# REPORT SUMMARY
# ============================================================

print("\n==============================")
print("BAD INCIDENT/DATE GROUPS")
print("==============================")
print(f"Total bad remove groups: {len(bad_remove_keys)}")

summary_rows = []

for remove_key, reasons in bad_day_reasons.items():
    for reason in reasons:
        summary_rows.append(
            {
                "remove_key": remove_key,
                "reason": reason,
            }
        )

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    print("\nDay-level removal reason counts:")
    print(summary_df["reason"].value_counts().sort_values(ascending=False))
else:
    print("\nNo day-level QAQC removals triggered.")

print("\n==============================")
print("BAD INCIDENT GROUPS")
print("==============================")
print(f"Total bad incidents: {len(bad_incidents)}")

incident_summary_rows = []

for incident, reasons in bad_incident_reasons.items():
    for reason in reasons:
        incident_summary_rows.append(
            {
                "incident": incident,
                "reason": reason,
            }
        )

if incident_summary_rows:
    incident_summary_df = pd.DataFrame(incident_summary_rows)
    print("\nIncident-level removal reason counts:")
    print(incident_summary_df["reason"].value_counts().sort_values(ascending=False))
else:
    print("\nNo incident-level QAQC removals triggered.")


# ============================================================
# WRITE OUTPUT GEOPACKAGES
# ============================================================

for output_path in [out_cleaned_gpkg, out_removed_gpkg]:
    if output_path.exists():
        output_path.unlink()

print("\nWriting cleaned GeoPackage")

for layer in layers:
    out = clean_for_output(cleaned[layer], original_crs)
    out.to_file(out_cleaned_gpkg, layer=layer, driver="GPKG")
    print(f"  {layer}: {len(out)} cleaned features")

print("\nWriting removed GeoPackage")

for layer in layers:
    if len(removed[layer]) > 0:
        rem_out = removed_for_output(removed[layer], original_crs)
        rem_out.to_file(out_removed_gpkg, layer=layer, driver="GPKG")
        print(f"  {layer}: {len(rem_out)} removed features")
    else:
        empty = removed_for_output(removed[layer], original_crs)
        empty.to_file(out_removed_gpkg, layer=layer, driver="GPKG")
        print(f"  {layer}: 0 removed features")

print("\nDone.")
print(f"Cleaned: {out_cleaned_gpkg}")
print(f"Removed: {out_removed_gpkg}")
