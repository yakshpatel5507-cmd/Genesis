import rasterio
import numpy as np
import torch
import mlstac
from pathlib import Path
from rasterio.transform import Affine


# ==================================================
# PATHS
# ==================================================

input_file = Path(
    "data/raw/sentinel2_utm_128_B02_B03_B04_B08.tif"
)

model_dir = Path(
    "models/SEN2SRLite_RGBN"
)

output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==================================================
# READ SENTINEL-2
# ==================================================

print("\nReading Sentinel-2 image...")

with rasterio.open(input_file) as src:

    image = src.read().astype(np.float32)

    original_transform = src.transform
    original_crs = src.crs

    print("Shape:", image.shape)
    print("CRS:", original_crs)
    print("Resolution:", src.res)
    print("Bounds:", src.bounds)


# ==================================================
# BAND ORDER
# ==================================================

# Downloaded image:
#
# Band 0 = B02
# Band 1 = B03
# Band 2 = B04
# Band 3 = B08
#
# SEN2SR expects:
#
# B04, B03, B02, B08

image = image[[2, 1, 0, 3]]

print("\nBand order changed to:")
print("B04, B03, B02, B08")


# ==================================================
# CLEAN DATA
# ==================================================

image = np.nan_to_num(
    image,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

# Sentinel Hub returned REFLECTANCE.
# Therefore values are already approximately 0–1.
#
# IMPORTANT:
# Do NOT divide by 10,000.


print("\nInput statistics:")

for i, name in enumerate(
    ["B04", "B03", "B02", "B08"]
):

    print(
        f"{name}: "
        f"min={image[i].min():.4f}, "
        f"max={image[i].max():.4f}, "
        f"mean={image[i].mean():.4f}"
    )


# ==================================================
# NUMPY → PYTORCH
# ==================================================

X = torch.from_numpy(image).float()

# [4, 128, 128]
#       ↓
# [1, 4, 128, 128]

X = X.unsqueeze(0).to(device)

print("\nModel input:")
print(X.shape)


# ==================================================
# LOAD SEN2SRLITE
# ==================================================

print("\nLoading SEN2SRLite...")

model = mlstac.load(
    str(model_dir)
).compiled_model(device=device)

model = model.to(device)
model.eval()

print("Model loaded successfully!")


# ==================================================
# SUPER RESOLUTION
# ==================================================

print("\nRunning SEN2SR...")
print("CPU inference may take some time.")

with torch.no_grad():

    superX = model(X)


print("\nSuper-resolution complete!")

print("Output tensor:", superX.shape)


# ==================================================
# PYTORCH → NUMPY
# ==================================================

output = (
    superX
    .squeeze(0)
    .cpu()
    .numpy()
    .astype(np.float32)
)

print("Output NumPy:", output.shape)


# ==================================================
# OUTPUT STATISTICS
# ==================================================

print("\nOutput statistics:")

for i, name in enumerate(
    ["B04", "B03", "B02", "B08"]
):

    print(
        f"{name}: "
        f"min={output[i].min():.4f}, "
        f"max={output[i].max():.4f}, "
        f"mean={output[i].mean():.4f}"
    )


# ==================================================
# SAVE RAW NUMPY
# ==================================================

npy_file = (
    output_dir /
    "sen2sr_utm_output.npy"
)

np.save(
    npy_file,
    output
)

print("\nNumPy output saved:")
print(npy_file)


# ==================================================
# CREATE 2.5 m TRANSFORM
# ==================================================

# Original:
#
# 10 m × 10 m
#
# SR:
#
# 2.5 m × 2.5 m
#
# because:
#
# 10 / 4 = 2.5


sr_transform = Affine(
    original_transform.a / 4,
    original_transform.b / 4,
    original_transform.c,

    original_transform.d / 4,
    original_transform.e / 4,
    original_transform.f
)


# ==================================================
# SAVE GEOTIFF
# ==================================================

output_file = (
    output_dir /
    "sen2sr_utm_2p5m.tif"
)


with rasterio.open(

    output_file,

    "w",

    driver="GTiff",

    height=output.shape[1],

    width=output.shape[2],

    count=output.shape[0],

    dtype="float32",

    crs=original_crs,

    transform=sr_transform,

    compress="deflate"

) as dst:

    dst.write(output)


print("\nGeoTIFF saved:")
print(output_file)


# ==================================================
# VERIFY OUTPUT
# ==================================================

with rasterio.open(output_file) as src:

    print("\n========== OUTPUT VERIFICATION ==========")

    print("Width:", src.width)
    print("Height:", src.height)
    print("Bands:", src.count)
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print("Bounds:", src.bounds)


print("\n==========================================")
print("SEN2SR UTM PIPELINE COMPLETE")
print("==========================================")