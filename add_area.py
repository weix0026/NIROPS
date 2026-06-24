import geopandas as gpd

# Read shapefile
gdf = gpd.read_file(r"C:\NIROPS_data\NIROPS_2025.gpkg", layer = 'Heat_Perimeter')

# Reproject to CONUS Albers (meters)
gdf_proj = gdf.to_crs("EPSG:5070")

# Area in hectares
gdf["AREA"] = gdf_proj.geometry.area / 10000

#gdf.to_file(r"C:\NIROPS_updated\fire_perimeters.gpkg", driver="GPKG")