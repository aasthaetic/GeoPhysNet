import xarray as xr
from pathlib import Path

emit_dir = Path("data/raw/emit")

files = list(emit_dir.glob("EMIT_L2A_RFL_*.nc"))

print("=" * 70)
print("EMIT REFLECTANCE DATA INSPECTION")
print("=" * 70)

print(f"\nReflectance files found: {len(files)}")

for file in files:
    print(f"  - {file.name}")

if not files:
    raise FileNotFoundError("No EMIT_L2A_RFL file found.")

file = files[0]

print("\n" + "=" * 70)
print("INSPECTING REFLECTANCE FILE")
print("=" * 70)

print(file)

# Open main dataset
ds = xr.open_dataset(file, engine="netcdf4")

print("\n" + "=" * 70)
print("MAIN DATASET")
print("=" * 70)

print(ds)

print("\n" + "=" * 70)
print("DIMENSIONS")
print("=" * 70)

for name, size in ds.sizes.items():
    print(f"{name}: {size}")

print("\n" + "=" * 70)
print("DATA VARIABLES")
print("=" * 70)

for name, variable in ds.data_vars.items():
    print(
        f"{name}: "
        f"dims={variable.dims}, "
        f"shape={variable.shape}, "
        f"dtype={variable.dtype}"
    )

print("\n" + "=" * 70)
print("OPENING LOCATION GROUP")
print("=" * 70)

location = xr.open_dataset(
    file,
    group="location",
    engine="netcdf4"
)

print(location)

print("\nLOCATION VARIABLES:")

for name, variable in location.data_vars.items():
    print(
        f"{name}: "
        f"dims={variable.dims}, "
        f"shape={variable.shape}, "
        f"dtype={variable.dtype}"
    )

print("\n" + "=" * 70)
print("OPENING SENSOR BAND PARAMETERS")
print("=" * 70)

bands = xr.open_dataset(
    file,
    group="sensor_band_parameters",
    engine="netcdf4"
)

print(bands)

print("\nBAND VARIABLES:")

for name, variable in bands.data_vars.items():
    print(
        f"{name}: "
        f"dims={variable.dims}, "
        f"shape={variable.shape}, "
        f"dtype={variable.dtype}"
    )

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

ds.close()
location.close()
bands.close()