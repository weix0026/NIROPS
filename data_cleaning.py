from add_area import gdf

area_by_day = list(gdf.groupby(["INCIDENTNA", "TRUEDATE"])["AREA"].sum().items())

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
    thr = -1
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
            if (not y) or (area - y[-1][1] >= thr):
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
print(clean(area_by_day))
print(number_of_fires(clean(area_by_day)))