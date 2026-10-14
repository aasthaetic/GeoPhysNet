import geopandas as gpd
import pandas as pd
import glob
import os

EMIT_FILE = "data/processed/gnn_dataset.csv"
NGDR_FOLDER = "data/raw/geology/ngdr_50k"
OUTPUT_FILE = "data/processed/gnn_dataset_ngdr.csv"

# 1. Load EMIT dataset
print("Loading EMIT dataset...")

emit = pd.read_csv(EMIT_FILE)

print("EMIT nodes:", len(emit))
print("EMIT columns:", list(emit.columns))

# 2. Convert EMIT points to GeoDataFrame
emit_gdf = gpd.GeoDataFrame(
    emit,
    geometry=gpd.points_from_xy(
        emit["longitude"],
        emit["latitude"]
    ),
    crs="EPSG:4326"
)

# 3. Load NGDR shapefiles
files = glob.glob(
    os.path.join(NGDR_FOLDER, "**", "*.shp"),
    recursive=True
)

print("\nNGDR shapefiles found:", len(files))

ngdr_list = []

for file in files:

    print("Loading:", file)

    gdf = gpd.read_file(file)

    gdf = gdf[
        [
            "lithologic",
            "group_name",
            "formation",
            "member",
            "age",
            "geometry"
        ]
    ]

    # Rename geological fields so they cannot clash
    gdf = gdf.rename(columns={
        "lithologic": "ngdr_lithology",
        "group_name": "ngdr_group",
        "formation": "ngdr_formation",
        "member": "ngdr_member",
        "age": "ngdr_age"
    })

    ngdr_list.append(gdf)

ngdr = gpd.GeoDataFrame(
    pd.concat(ngdr_list, ignore_index=True),
    crs="EPSG:4326"
)

print("\nTotal NGDR polygons:", len(ngdr))

# 4. Spatial join
print("\nPerforming spatial join...")

joined = gpd.sjoin(
    emit_gdf,
    ngdr,
    how="left",
    predicate="within"
)

# 5. Check result
print("\nJoined columns:")
print(list(joined.columns))

# 6. Remove geometry/index columns
joined = joined.drop(
    columns=["geometry", "index_right"],
    errors="ignore"
)

# 7. Save
joined.to_csv(
    OUTPUT_FILE,
    index=False
)

# 8. Summary
total = len(joined)

labeled = joined["ngdr_lithology"].notna().sum()

print("\n-----------------------------")
print("SPATIAL JOIN COMPLETE")
print("-----------------------------")

print("Total EMIT nodes:", total)
print("Nodes with NGDR lithology:", labeled)

print(
    "Percentage labeled:",
    round((labeled / total) * 100, 2),
    "%"
)

print("\nNGDR lithology distribution:")
print(
    joined["ngdr_lithology"]
    .value_counts(dropna=False)
)

print("\nSaved to:")
print(OUTPUT_FILE)