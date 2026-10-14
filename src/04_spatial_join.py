import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

emit_file = Path("data/interim/emit_features.csv")

geology_file = Path(
    "data/raw/geology/ngdr_lithology_polygons.geojson"
)

output_file = Path(
    "data/interim/emit_geology.csv"
)


print("=" * 70)
print("EMIT + NGDR GEOLOGY SPATIAL JOIN")
print("=" * 70)


# ============================================================
# LOAD EMIT DATA
# ============================================================

print("\nLoading EMIT features...")

emit = pd.read_csv(emit_file)

print(f"EMIT pixels loaded: {len(emit):,}")


# ============================================================
# LOAD GEOLOGY
# ============================================================

print("\nLoading NGDR geology...")

geology = gpd.read_file(geology_file)

print(f"Geology polygons: {len(geology):,}")
print(f"CRS reported by file: {geology.crs}")

print("\nReported geology bounds:")
print(geology.total_bounds)


# ============================================================
# CORRECT GEOLOGY CRS
# ============================================================

# The coordinates are in metres and have Web Mercator-scale
# values, although the file reports EPSG:4326.
# Therefore we override the incorrect CRS metadata.

geology = geology.set_crs(
    "EPSG:3857",
    allow_override=True
)

print("\nUsing geology CRS: EPSG:3857")


# ============================================================
# CONVERT GEOLOGY TO WGS84
# ============================================================

geology_wgs84 = geology.to_crs("EPSG:4326")

min_lon, min_lat, max_lon, max_lat = (
    geology_wgs84.total_bounds
)

print("\nGeology geographic extent:")
print(f"Longitude: {min_lon:.6f} to {max_lon:.6f}")
print(f"Latitude:  {min_lat:.6f} to {max_lat:.6f}")


# ============================================================
# FILTER EMIT PIXELS
# ============================================================

print("\nFiltering EMIT pixels to geology extent...")

emit_filtered = emit[
    (emit["longitude"] >= min_lon) &
    (emit["longitude"] <= max_lon) &
    (emit["latitude"] >= min_lat) &
    (emit["latitude"] <= max_lat)
].copy()

print(
    f"EMIT pixels before filtering: "
    f"{len(emit):,}"
)

print(
    f"EMIT pixels after filtering:  "
    f"{len(emit_filtered):,}"
)


# ============================================================
# CREATE EMIT POINTS
# ============================================================

print("\nCreating EMIT spatial points...")

geometry = [
    Point(lon, lat)
    for lon, lat in zip(
        emit_filtered["longitude"],
        emit_filtered["latitude"]
    )
]

emit_gdf = gpd.GeoDataFrame(
    emit_filtered,
    geometry=geometry,
    crs="EPSG:4326"
)


# ============================================================
# PROJECT EMIT POINTS
# ============================================================

print("Projecting EMIT points to EPSG:3857...")

emit_gdf = emit_gdf.to_crs("EPSG:3857")


# ============================================================
# SPATIAL JOIN
# ============================================================

print("\nPerforming spatial join...")

joined = gpd.sjoin(
    emit_gdf,
    geology[
        [
            "lithologic",
            "age",
            "supergroup",
            "group_name",
            "formation",
            "member",
            "geometry"
        ]
    ],
    how="inner",
    predicate="within"
)

print(
    f"Raw spatial matches: {len(joined):,}"
)


# ============================================================
# REMOVE DUPLICATE PIXELS
# ============================================================

print("\nRemoving duplicate pixel matches...")

joined = joined.drop_duplicates(
    subset=["node_id"]
)


# ============================================================
# REMOVE GEOMETRY COLUMNS
# ============================================================

for column in ["geometry", "index_right"]:
    if column in joined.columns:
        joined = joined.drop(columns=column)


# ============================================================
# SAVE
# ============================================================

print("\nSaving joined dataset...")

joined.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SPATIAL JOIN COMPLETE")
print("=" * 70)

print(f"\nOutput file:")
print(output_file)

print(f"\nLabeled EMIT pixels: {len(joined):,}")

if len(emit_filtered) > 0:
    percentage = (
        len(joined) / len(emit_filtered)
    ) * 100

    print(
        f"Percentage of filtered pixels labeled: "
        f"{percentage:.2f}%"
    )

print("\nLithology distribution:")

if "lithologic" in joined.columns:
    print(
        joined["lithologic"]
        .value_counts()
        .head(20)
    )

print("\nColumns:")

for column in joined.columns:
    print(f"  - {column}")

print("\nDone.")