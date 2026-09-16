import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


file_path = "data/raw/sentinel2_test_B02_B03_B04_B08.tif"

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)


with rasterio.open(file_path) as src:

    # Our downloaded order:
    # Band 1 = B02 Blue
    # Band 2 = B03 Green
    # Band 3 = B04 Red
    # Band 4 = B08 NIR

    blue = src.read(1)
    green = src.read(2)
    red = src.read(3)
    nir = src.read(4)


def stretch(image):
    """
    Improve visualization by clipping extreme values
    and scaling reflectance to 0-1.
    """

    low, high = np.percentile(image, (2, 98))

    image = np.clip(image, low, high)

    image = (image - low) / (high - low)

    return image


# --------------------------------------------------
# RGB IMAGE
# --------------------------------------------------

rgb = np.dstack([
    stretch(red),
    stretch(green),
    stretch(blue)
])

plt.figure(figsize=(8, 8))
plt.imshow(rgb)
plt.title("Sentinel-2 RGB - 10m")
plt.axis("off")

rgb_path = output_dir / "sentinel2_test_RGB.png"
plt.savefig(rgb_path, dpi=150, bbox_inches="tight")
plt.show()

print("RGB saved to:", rgb_path)


# --------------------------------------------------
# NIR IMAGE
# --------------------------------------------------

nir_display = stretch(nir)

plt.figure(figsize=(8, 8))
plt.imshow(nir_display, cmap="gray")
plt.title("Sentinel-2 NIR (B08) - 10m")
plt.axis("off")

nir_path = output_dir / "sentinel2_test_NIR.png"
plt.savefig(nir_path, dpi=150, bbox_inches="tight")
plt.show()

print("NIR saved to:", nir_path)