import rasterio
import numpy as np
import torch
import mlstac
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

input_file = Path(
    "data/raw/sentinel2_128_B02_B03_B04_B08.tif"
)

model_dir = Path(
    "models/SEN2SRLite_RGBN"
)

output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Read Sentinel-2 image
# --------------------------------------------------

print("\nReading Sentinel-2 image...")

with rasterio.open(input_file) as src:

    image = src.read().astype(np.float32)

    print("Original shape:", image.shape)
    print("CRS:", src.crs)
    print("Resolution:", src.res)

    # Save metadata for later GeoTIFF creation
    transform = src.transform
    crs = src.crs


# --------------------------------------------------
# Current order:
#
# image[0] = B02
# image[1] = B03
# image[2] = B04
# image[3] = B08
#
# SEN2SR expects:
#
# B04, B03, B02, B08
# --------------------------------------------------

image = image[[2, 1, 0, 3]]

print("Reordered shape:", image.shape)


# --------------------------------------------------
# Clean invalid values
# --------------------------------------------------

image = np.nan_to_num(
    image,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

# Our Process API already returned REFLECTANCE
# values in approximately 0-1.
#
# Therefore:
# DO NOT divide by 10,000 here.


# --------------------------------------------------
# Convert NumPy → PyTorch
# --------------------------------------------------

X = torch.from_numpy(image).float()

# Add batch dimension:
#
# [4, 128, 128]
#       ↓
# [1, 4, 128, 128]

X = X.unsqueeze(0).to(device)

print("Model input shape:", X.shape)


# --------------------------------------------------
# Load pretrained SEN2SRLite model
# --------------------------------------------------

print("\nLoading SEN2SR model...")

model = mlstac.load(
    str(model_dir)
).compiled_model(device=device)

model = model.to(device)

model.eval()

print("Model loaded successfully!")


# --------------------------------------------------
# Run super-resolution
# --------------------------------------------------

print("\nRunning super-resolution...")
print("This may take a little while on CPU...")

with torch.no_grad():

    superX = model(X)

print("\nSuper-resolution complete!")

print("Output tensor shape:", superX.shape)


# --------------------------------------------------
# Convert output to NumPy
# --------------------------------------------------

output = superX.squeeze(0).cpu().numpy()

print("Output NumPy shape:", output.shape)


# --------------------------------------------------
# Save raw output
# --------------------------------------------------

raw_output = output.astype(np.float32)

raw_file = output_dir / "sen2sr_output.npy"

np.save(raw_file, raw_output)

print("\nRaw output saved:")
print(raw_file)


# --------------------------------------------------
# Save output as GeoTIFF
# --------------------------------------------------

# Input: 128 × 128
# Output: 512 × 512
#
# Therefore pixel size becomes 1/4.
#
# Example:
# 10 m → 2.5 m
#
# For our EPSG:4326 test:
# degree resolution is divided by 4.

new_transform = transform * transform.scale(
    1 / 4,
    1 / 4
)

output_file = output_dir / "sen2sr_output_2p5m.tif"

with rasterio.open(
    output_file,
    "w",
    driver="GTiff",
    height=output.shape[1],
    width=output.shape[2],
    count=output.shape[0],
    dtype="float32",
    crs=crs,
    transform=new_transform,
) as dst:

    dst.write(raw_output)


print("\nGeoTIFF saved:")
print(output_file)

print("\nDONE!")