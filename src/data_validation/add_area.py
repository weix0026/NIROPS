import geopandas as gpd

# Read shapefile
gdf = gpd.read_file(r"D:\MasterData\NIROPS_GPKG\NIROPS_GPKG\NIROPS_LEOTEST\NIROPS_2025.gpkg", layer = 'Heat_Perimeter')

# Reproject to CONUS Albers (meters)
gdf_proj = gdf.to_crs("EPSG:5070")

# Area in hectares
gdf["AREA_ha"] = gdf_proj.geometry.area / 10000

gdf.to_file(r"D:\MasterData\NIROPS_GPKG\NIROPS_GPKG\NIROPS_LEOTEST\NIROPS_2025.gpkg", layer = 'Heat_Perimeter_with_ha', driver="GPKG")
