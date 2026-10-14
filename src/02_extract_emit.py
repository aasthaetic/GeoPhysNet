import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

emit_dir = Path("data/raw/emit")
output_dir = Path("data/interim")
output_dir.mkdir(parents=True, exist_ok=True)

reflectance_file = list(
    emit_dir.glob("EMIT_L2A_RFL_*.nc")
)[0]

output_file = output_dir / "emit_features.csv"


# ============================================================
# TARGET WAVELENGTHS
# ============================================================

target_wavelengths = np.array([
    450,
    550,
    650,
    700,
    800,
    900,
    1000,
    1200,
    1600,
    1800,
    2000,
    2200,
    2300,
    2400
])


print("=" * 70)
print("EMIT RAW DATA EXTRACTION")
print("=" * 70)

print(f"\nInput file:")
print(reflectance_file)


# ============================================================
# OPEN DATASETS
# ============================================================

ds = xr.open_dataset(
    reflectance_file,
    engine="netcdf4"
)

location = xr.open_dataset(
    reflectance_file,
    group="location",
    engine="netcdf4"
)

bands = xr.open_dataset(
    reflectance_file,
    group="sensor_band_parameters",
    engine="netcdf4"
)


# ============================================================
# READ WAVELENGTHS
# ============================================================

wavelengths = bands["wavelengths"].values

print(f"\nTotal spectral bands: {len(wavelengths)}")

print("\nSelected wavelengths:")

selected_indices = []

for target in target_wavelengths:

    index = int(
        np.abs(wavelengths - target).argmin()
    )

    actual = wavelengths[index]

    selected_indices.append(index)

    print(
        f"Requested: {target:4d} nm "
        f"-> Actual: {actual:.2f} nm "
        f"(band {index})"
    )


# ============================================================
# LOAD LOCATION DATA
# ============================================================

print("\nLoading latitude, longitude and elevation...")

lat = location["lat"].values
lon = location["lon"].values
elev = location["elev"].values


# ============================================================
# LOAD SELECTED REFLECTANCE BANDS
# ============================================================

print("Loading selected reflectance bands...")

reflectance = ds["reflectance"].isel(
    bands=selected_indices
).values


print(
    f"Reflectance array shape: {reflectance.shape}"
)


# ============================================================
# CONVERT TO PIXEL TABLE
# ============================================================

print("\nConverting pixels to table...")

n_rows, n_cols, n_bands = reflectance.shape

lat_flat = lat.reshape(-1)
lon_flat = lon.reshape(-1)
elev_flat = elev.reshape(-1)

reflectance_flat = reflectance.reshape(
    -1,
    n_bands
)


data = {
    "latitude": lat_flat,
    "longitude": lon_flat,
    "elevation_m": elev_flat
}


for i, target in enumerate(target_wavelengths):

    actual = wavelengths[selected_indices[i]]

    column_name = (
        f"reflectance_{int(round(actual))}nm"
    )

    data[column_name] = reflectance_flat[:, i]


df = pd.DataFrame(data)


# ============================================================
# REMOVE INVALID PIXELS
# ============================================================

print("\nRemoving invalid pixels...")

before = len(df)

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=[
        "latitude",
        "longitude"
    ]
)

reflectance_columns = [
    column
    for column in df.columns
    if column.startswith("reflectance_")
]

df = df.dropna(
    subset=reflectance_columns,
    how="all"
)

after = len(df)


print(f"Pixels before filtering: {before:,}")
print(f"Pixels after filtering:  {after:,}")
print(f"Pixels removed:          {before - after:,}")


# ============================================================
# ADD NODE ID
# ============================================================

df.insert(
    0,
    "node_id",
    np.arange(len(df))
)


# ============================================================
# SAVE
# ============================================================

print("\nSaving extracted dataset...")

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)

print(f"\nOutput file:")
print(output_file)

print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
for column in df.columns:
    print(f"  - {column}")

print("\nFirst 5 rows:")
print(df.head())

print("\nLatitude range:")
print(
    f"{df['latitude'].min():.6f} "
    f"to "
    f"{df['latitude'].max():.6f}"
)

print("\nLongitude range:")
print(
    f"{df['longitude'].min():.6f} "
    f"to "
    f"{df['longitude'].max():.6f}"
)

print("\nElevation range:")
print(
    f"{df['elevation_m'].min():.2f} m "
    f"to "
    f"{df['elevation_m'].max():.2f} m"
)

print("\nDone.")


# ============================================================
# CLOSE FILES
# ============================================================

ds.close()
location.close()
bands.close()