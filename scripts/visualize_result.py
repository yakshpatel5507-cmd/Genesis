import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# Files
# --------------------------------------------------
input_file = "data/raw/sentinel2_utm_128_B02_B03_B04_B08.tif"

output_file = "results/sen2sr_utm_2p5m.tif"


# --------------------------------------------------
# Read original Sentinel-2 image
# --------------------------------------------------

with rasterio.open(input_file) as src:

    original = src.read().astype(np.float32)

print("Original:", original.shape)


# Original order:
# B02, B03, B04, B08

B02 = original[0]
B03 = original[1]
B04 = original[2]
B08 = original[3]


# --------------------------------------------------
# Read SEN2SR output
# --------------------------------------------------

with rasterio.open(output_file) as src:

    sr = src.read().astype(np.float32)

print("Super-resolved:", sr.shape)


# Output order:
# B04, B03, B02, B08

sr_B04 = sr[0]
sr_B03 = sr[1]
sr_B02 = sr[2]
sr_B08 = sr[3]


# --------------------------------------------------
# RGB images
# --------------------------------------------------

rgb_original = np.stack(
    [B04, B03, B02],
    axis=-1
)

rgb_sr = np.stack(
    [sr_B04, sr_B03, sr_B02],
    axis=-1
)


# --------------------------------------------------
# Percentile stretch
#
# Satellite reflectance values are not naturally
# displayed like normal photographs.
# This makes the images easier to see.
# --------------------------------------------------

def stretch(image):

    low = np.percentile(image, 2)
    high = np.percentile(image, 98)

    image = (image - low) / (high - low + 1e-8)

    return np.clip(image, 0, 1)


rgb_original_display = stretch(rgb_original)
rgb_sr_display = stretch(rgb_sr)


# --------------------------------------------------
# RGB comparison
# --------------------------------------------------

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.imshow(rgb_original_display)
plt.title("Original Sentinel-2 — 10 m")
plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(rgb_sr_display)
plt.title("SEN2SR — 2.5 m")
plt.axis("off")


plt.tight_layout()

comparison_file = Path(
    "results/rgb_comparison.png"
)

plt.savefig(
    comparison_file,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nRGB comparison saved:")
print(comparison_file)


# --------------------------------------------------
# NIR comparison
# --------------------------------------------------

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.imshow(B08, cmap="gray")
plt.title("Original NIR — 10 m")
plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(sr_B08, cmap="gray")
plt.title("SEN2SR NIR — 2.5 m")
plt.axis("off")


plt.tight_layout()

nir_file = Path(
    "results/nir_comparison.png"
)

plt.savefig(
    nir_file,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("NIR comparison saved:")
print(nir_file)


# --------------------------------------------------
# NDVI
# --------------------------------------------------

ndvi_original = (
    (B08 - B04) /
    (B08 + B04 + 1e-8)
)

ndvi_sr = (
    (sr_B08 - sr_B04) /
    (sr_B08 + sr_B04 + 1e-8)
)


# --------------------------------------------------
# NDVI comparison
# --------------------------------------------------

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.imshow(
    ndvi_original,
    cmap="RdYlGn",
    vmin=-1,
    vmax=1
)

plt.title("Original NDVI — 10 m")
plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    ndvi_sr,
    cmap="RdYlGn",
    vmin=-1,
    vmax=1
)

plt.title("SEN2SR NDVI — 2.5 m")
plt.axis("off")


plt.tight_layout()

ndvi_file = Path(
    "results/ndvi_comparison.png"
)

plt.savefig(
    ndvi_file,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("NDVI comparison saved:")
print(ndvi_file)


# --------------------------------------------------
# Basic statistics
# --------------------------------------------------

print("\n========== STATISTICS ==========")

print("\nOriginal NDVI:")
print("Min :", ndvi_original.min())
print("Max :", ndvi_original.max())
print("Mean:", ndvi_original.mean())

print("\nSR NDVI:")
print("Min :", ndvi_sr.min())
print("Max :", ndvi_sr.max())
print("Mean:", ndvi_sr.mean())