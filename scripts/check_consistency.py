import rasterio
import numpy as np
from pathlib import Path
from skimage.transform import resize


# --------------------------------------------------
# Files
# --------------------------------------------------

input_file = Path(
    "data/raw/sentinel2_128_B02_B03_B04_B08.tif"
)

sr_file = Path(
    "results/sen2sr_output_2p5m.tif"
)


# --------------------------------------------------
# Read original
# --------------------------------------------------

with rasterio.open(input_file) as src:

    original = src.read().astype(np.float32)

print("Original shape:", original.shape)


# Original:
# B02, B03, B04, B08


# --------------------------------------------------
# Read SR output
# --------------------------------------------------

with rasterio.open(sr_file) as src:

    sr = src.read().astype(np.float32)

print("SR shape:", sr.shape)


# SR:
# B04, B03, B02, B08


# Reorder SR back to:
# B02, B03, B04, B08

sr = sr[[2, 1, 0, 3]]


# --------------------------------------------------
# Downsample SR from 512 → 128
# --------------------------------------------------

sr_downsampled = np.zeros_like(original)


for band in range(4):

    sr_downsampled[band] = resize(
        sr[band],
        (128, 128),
        order=1,
        preserve_range=True,
        anti_aliasing=True
    ).astype(np.float32)


print(
    "Downsampled SR shape:",
    sr_downsampled.shape
)


# --------------------------------------------------
# Metrics
# --------------------------------------------------

band_names = [
    "B02 Blue",
    "B03 Green",
    "B04 Red",
    "B08 NIR"
]


print("\n========================================")
print("SPECTRAL / RADIOMETRIC CONSISTENCY")
print("========================================")


for i, name in enumerate(band_names):

    ref = original[i]
    pred = sr_downsampled[i]

    mae = np.mean(
        np.abs(ref - pred)
    )

    rmse = np.sqrt(
        np.mean((ref - pred) ** 2)
    )

    correlation = np.corrcoef(
        ref.flatten(),
        pred.flatten()
    )[0, 1]

    print(f"\n{name}")

    print(f"  MAE         : {mae:.6f}")
    print(f"  RMSE        : {rmse:.6f}")
    print(f"  Correlation : {correlation:.6f}")


# --------------------------------------------------
# NDVI consistency
# --------------------------------------------------

original_red = original[2]
original_nir = original[3]

sr_red = sr_downsampled[2]
sr_nir = sr_downsampled[3]


original_ndvi = (
    (original_nir - original_red) /
    (original_nir + original_red + 1e-8)
)

sr_ndvi = (
    (sr_nir - sr_red) /
    (sr_nir + sr_red + 1e-8)
)


ndvi_mae = np.mean(
    np.abs(original_ndvi - sr_ndvi)
)

ndvi_rmse = np.sqrt(
    np.mean(
        (original_ndvi - sr_ndvi) ** 2
    )
)

ndvi_correlation = np.corrcoef(
    original_ndvi.flatten(),
    sr_ndvi.flatten()
)[0, 1]


print("\n========================================")
print("NDVI CONSISTENCY")
print("========================================")

print(f"NDVI MAE         : {ndvi_mae:.6f}")
print(f"NDVI RMSE        : {ndvi_rmse:.6f}")
print(f"NDVI Correlation : {ndvi_correlation:.6f}")


# --------------------------------------------------
# Difference map
# --------------------------------------------------

difference = np.mean(
    np.abs(original - sr_downsampled),
    axis=0
)


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 7))

plt.imshow(
    difference,
    cmap="magma"
)

plt.colorbar(
    label="Mean Absolute Spectral Difference"
)

plt.title(
    "Spectral Consistency Difference Map"
)

plt.axis("off")

plt.tight_layout()

difference_file = Path(
    "results/spectral_difference_map.png"
)

plt.savefig(
    difference_file,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(
    "\nDifference map saved:",
    difference_file
)