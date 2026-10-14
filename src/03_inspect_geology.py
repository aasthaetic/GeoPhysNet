import geopandas as gpd
from pathlib import Path

print("=" * 70)
print("NGDR GEOLOGY DATA INSPECTION")
print("=" * 70)

geology_dir = Path("data/raw/geology")

files = list(geology_dir.glob("*.geojson")) + list(geology_dir.glob("*.gpkg"))

print(f"\nGeology files found: {len(files)}")

for file in files:
    print(f"  - {file.name}")

if not files:
    raise FileNotFoundError(
        "No GeoJSON or GeoPackage found in data/raw/geology"
    )

file = files[0]

print("\n" + "=" * 70)
print("READING GEOLOGY FILE")
print("=" * 70)

print(file)

gdf = gpd.read_file(file)

print("\n" + "=" * 70)
print("GEOLOGY DATASET")
print("=" * 70)

print(gdf)

print("\n" + "=" * 70)
print("SHAPE")
print("=" * 70)

print(f"Rows: {len(gdf):,}")
print(f"Columns: {len(gdf.columns)}")

print("\n" + "=" * 70)
print("CRS")
print("=" * 70)

print(gdf.crs)

print("\n" + "=" * 70)
print("BOUNDS")
print("=" * 70)

print(gdf.total_bounds)

print("\n" + "=" * 70)
print("COLUMNS")
print("=" * 70)

for column in gdf.columns:
    print(f"  - {column}")

print("\n" + "=" * 70)
print("GEOMETRY TYPES")
print("=" * 70)

print(gdf.geometry.geom_type.value_counts())

print("\n" + "=" * 70)
print("SAMPLE ATTRIBUTES")
print("=" * 70)

print(gdf.head())

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)