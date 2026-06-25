from src.data_validation.add_area import gdf

# OLD CODE: grouped only by incident and date, so different times on the same day were combined.
# area_by_day = list(gdf.groupby(["INCIDENTNA", "TRUEDATE"])["AREA"].sum().items())

# NEW CODE: use the AREA_ha column created in add_area.py.
area_by_day = list(gdf.groupby(["INCIDENTNA", "TRUEDATE", "TRUETIME"])["AREA_ha"].sum().items())

print(area_by_day)
print(len(area_by_day))

def group_by_incident(l: list, ict: str):
    ict_lst = []
    for fire in l:
        if fire[0][0] == ict:
            ict_lst.append(fire)
        else:
            pass
    return ict_lst

def clean(l: list):
    grouped_list = []
    clean_list = []
    # OLD CODE: allowed the next day to be smaller by 1 AREA unit.
    # thr = -1

    # NEW CODE: day i+1 should be at least the same size as day t.
    thr = 0
    for fire in l:
        ict_lst = group_by_incident(l, fire[0][0])
        if ict_lst not in grouped_list:
            grouped_list.append(ict_lst)
        else:
            pass
    for ict in grouped_list:
        y = []
        for item in ict:
            area = item[1]
            # OLD CODE: this silently skipped/deleted rows when area decreased.
            # if (not y) or (area - y[-1][1] >= thr):
            #     y.append(item)

            # NEW CODE: report shrinking records before skipping them.
            if y and area - y[-1][1] < thr:
                previous_time = y[-1][0][2] if len(y[-1][0]) > 2 else ""
                current_time = item[0][2] if len(item[0]) > 2 else ""
                print(
                    "ERROR: fire area shrank",
                    item[0][0],
                    "from",
                    y[-1][0][1],
                    previous_time,
                    "to",
                    item[0][1],
                    current_time,
                    "area:",
                    y[-1][1],
                    "->",
                    area,
                )
                continue
            y.append(item)
        clean_list.append(y)
    return clean_list

def number_of_fires(l: list[list]):
    length = 0
    for item in l:
        for val in item:
            length += 1
    return length

lst = [(('2025_Bear_Gulch_WAOLF000178', '20250709'),312.4067301486287),
       (('2025_Bear_Gulch_WAOLF000178', '20250711'), 312.0934829023071),
       (('2025_Bear_Gulch_WAOLF000178', '20250713'), 359.5260225963558),
       (('2025_Bear_Gulch_WAOLF000178', '20250715'), 180.08764796177337),
       (('2025_Bear_Gulch_WAOLF000178', '20250717'), 428.74488720393947),
       (('2025_Bear_Gulch_WAOLF000178', '20250718'), 256.0937775521457),
       (('2025_Bear_Gulch_WAOLF000178', '20250720'), 540.6133349960891),
       (('2025_Bear_Gulch_WAOLF000178', '20250721'), 308.22570334700595),
       (('2025_Bear_Gulch_WAOLF000178', '20250723'), 323.49709654410395),
       (('2025_Bear_Gulch_WAOLF000178', '20250724'), 373.4449353601931),
       (('2025_Bear_Gulch_WAOLF000178', '20250725'), 373.4392878522099),
       (('2025_Bear_Gulch_WAOLF000178', '20250727'), 849.7157603482165),
       (('2025_Bear_Gulch_WAOLF000178', '20250729'), 834.7459371561665),
       (('2025_Bear_Gulch_WAOLF000178', '20250730'), 1262.944213381109),
       (('2025_Bear_Gulch_WAOLF000178', '20250731'), 1613.0688484510895),
       (('2025_Bear_Gulch_WAOLF000178', '20250802'), 3747.6233716313895),
       (('2025_Bear_Gulch_WAOLF000178', '20250808'), 4480.214302990491),
       (('2025_Bear_Gulch_WAOLF000178', '20250809'), 2262.1856915195654),
       (('2025_Bear_Gulch_WAOLF000178', '20250810'), 2287.275192120252),
       (('2025_Bear_Gulch_WAOLF000178', '20251016'), 8187.96926324397)]
# OLD CODE: called clean(area_by_day) twice, so error messages printed twice.
# print(clean(area_by_day))
# print(number_of_fires(clean(area_by_day)))

# NEW CODE: call clean once, then reuse the result.
cleaned_area_by_day = clean(area_by_day)
print(cleaned_area_by_day)
print(number_of_fires(cleaned_area_by_day))

# NEW CODE: write only the non-shrinking time steps to a new GeoPackage layer.
kept_time_steps = set()
for incident in cleaned_area_by_day:
    for item in incident:
        kept_time_steps.add(item[0])

cleaned_gdf = gdf[
    gdf.set_index(["INCIDENTNA", "TRUEDATE", "TRUETIME"]).index.isin(kept_time_steps)
]

cleaned_gdf.to_file(
    r"D:\MasterData\NIROPS_GPKG\NIROPS_GPKG\NIROPS_LEOTEST\NIROPS_2025.gpkg",
    layer="Heat_Perimeter_non_shrinking",
    driver="GPKG",
)

print("Wrote non-shrinking perimeters to layer: Heat_Perimeter_non_shrinking")
