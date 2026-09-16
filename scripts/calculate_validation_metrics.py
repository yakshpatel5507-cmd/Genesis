import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skimage.metrics import structural_similarity as ssim


# ============================================================
# FILE PATHS
# ============================================================

SR_PATH = "results/ROI_0000_SEN2SR_SR.tif"
HR_PATH = "data/validation/ROI_0000/hr.tif"
LR_PATH = "results/ROI_0000_degraded_LR.tif"


# ============================================================
# LOAD TIFF
# ============================================================

def load_tif(path):
    with rasterio.open(path) as src:
        data = src.read()
        profile = src.profile

    return data, profile


# ============================================================
# METRICS FUNCTION
# ============================================================

def calculate_metrics(pred, truth):

    # Flatten for pixel-wise metrics
    pred_flat = pred.flatten()
    truth_flat = truth.flatten()

    # -------------------------
    # MAE
    # -------------------------
    mae = mean_absolute_error(
        truth_flat,
        pred_flat
    )

    # -------------------------
    # RMSE
    # -------------------------
    rmse = np.sqrt(
        mean_squared_error(
            truth_flat,
            pred_flat
        )
    )

    # -------------------------
    # PSNR
    #
    # Images are normalized to 0-1,
    # so MAX_I = 1
    # -------------------------
    if rmse == 0:
        psnr = float("inf")
    else:
        psnr = 20 * np.log10(1.0 / rmse)

    # -------------------------
    # SSIM
    #
    # Only calculate for a
    # single 2D band.
    #
    # For a 4-band image, pred
    # has shape (4,H,W), which
    # should NOT be passed directly
    # to skimage SSIM.
    # -------------------------
    if pred.ndim == 2:

        structural = ssim(
            truth,
            pred,
            data_range=1.0
        )

    else:

        structural = None

    # -------------------------
    # Correlation
    # -------------------------
    if np.std(pred_flat) == 0 or np.std(truth_flat) == 0:
        corr = np.nan
    else:
        corr = np.corrcoef(
            truth_flat,
            pred_flat
        )[0, 1]

    return mae, rmse, psnr, structural, corr


# ============================================================
# PRINT BAND METRICS
# ============================================================

def print_band_metrics(name, pred, truth):

    mae, rmse, psnr, structural, corr = calculate_metrics(
        pred,
        truth
    )

    print(
        f"{name:<6} | "
        f"MAE: {mae:.6f} | "
        f"RMSE: {rmse:.6f} | "
        f"PSNR: {psnr:.3f} dB | "
        f"SSIM: {structural:.6f} | "
        f"Corr: {corr:.6f}"
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": structural,
        "corr": corr
    }


# ============================================================
# LOAD DATA
# ============================================================

sr, sr_profile = load_tif(SR_PATH)
hr, hr_profile = load_tif(HR_PATH)
lr, lr_profile = load_tif(LR_PATH)


print("SR:", sr.shape)
print("HR:", hr.shape)
print("LR:", lr.shape)


# ============================================================
# NORMALIZATION
# ============================================================

# SEN2SR output is already approximately 0-1
sr = sr.astype(np.float32)

# NAIP HR is uint8 [0,255]
hr = hr.astype(np.float32) / 255.0

# OpenSR degraded LR is already 0-1
lr = lr.astype(np.float32)


# ============================================================
# SAFETY CHECK
# ============================================================

print("\nData ranges:")
print(
    f"SR: min={sr.min():.6f}, max={sr.max():.6f}"
)

print(
    f"HR: min={hr.min():.6f}, max={hr.max():.6f}"
)

print(
    f"LR: min={lr.min():.6f}, max={lr.max():.6f}"
)


# ============================================================
# BAND NAMES
#
# SEN2SR RGBN input:
# Red, Green, Blue, NIR
# ============================================================

band_names = [
    "Red",
    "Green",
    "Blue",
    "NIR"
]


# ============================================================
# SEN2SR vs HR
# ============================================================

print("\n" + "=" * 40)
print("SEN2SR vs NAIP HR")
print("=" * 40)


sr_results = []

for i, band_name in enumerate(band_names):

    result = print_band_metrics(
        band_name,
        sr[i],
        hr[i]
    )

    sr_results.append(result)


# ============================================================
# OVERALL SEN2SR METRICS
# ============================================================

overall_sr = calculate_metrics(
    sr,
    hr
)

print(
    f"\nOverall | "
    f"MAE: {overall_sr[0]:.6f} | "
    f"RMSE: {overall_sr[1]:.6f} | "
    f"PSNR: {overall_sr[2]:.3f} dB | "
    f"Corr: {overall_sr[4]:.6f}"
)


# ============================================================
# NDVI - SEN2SR
#
# RGBN order:
# Red = band 0
# NIR = band 3
#
# NDVI = (NIR - Red)/(NIR + Red)
# ============================================================

def calculate_ndvi(red, nir):

    denominator = nir + red

    ndvi = np.zeros_like(red)

    valid = np.abs(denominator) > 1e-8

    ndvi[valid] = (
        (nir[valid] - red[valid])
        / denominator[valid]
    )

    return ndvi


sr_ndvi = calculate_ndvi(
    sr[0],
    sr[3]
)

hr_ndvi = calculate_ndvi(
    hr[0],
    hr[3]
)


# NDVI metrics
ndvi_mae = mean_absolute_error(
    hr_ndvi.flatten(),
    sr_ndvi.flatten()
)

ndvi_rmse = np.sqrt(
    mean_squared_error(
        hr_ndvi.flatten(),
        sr_ndvi.flatten()
    )
)

ndvi_corr = np.corrcoef(
    hr_ndvi.flatten(),
    sr_ndvi.flatten()
)[0, 1]


print("\nNDVI:")
print(
    f"MAE: {ndvi_mae:.6f} | "
    f"RMSE: {ndvi_rmse:.6f} | "
    f"Corr: {ndvi_corr:.6f}"
)


# ============================================================
# BICUBIC UPSAMPLING
# ============================================================

print("\n" + "=" * 40)
print("Creating Bicubic Baseline")
print("=" * 40)


# Convert numpy -> torch
lr_tensor = torch.from_numpy(lr).unsqueeze(0)


# Bicubic 4x upsampling
bicubic_tensor = F.interpolate(
    lr_tensor,
    size=(hr.shape[1], hr.shape[2]),
    mode="bicubic",
    align_corners=False
)


bicubic = bicubic_tensor.squeeze(0).numpy()


print(
    "Bicubic shape:",
    bicubic.shape
)


# ============================================================
# BICUBIC vs HR
# ============================================================

print("\n" + "=" * 40)
print("BICUBIC vs NAIP HR")
print("=" * 40)


bicubic_results = []

for i, band_name in enumerate(band_names):

    result = print_band_metrics(
        band_name,
        bicubic[i],
        hr[i]
    )

    bicubic_results.append(result)


# ============================================================
# OVERALL BICUBIC METRICS
# ============================================================

overall_bicubic = calculate_metrics(
    bicubic,
    hr
)

print(
    f"\nOverall | "
    f"MAE: {overall_bicubic[0]:.6f} | "
    f"RMSE: {overall_bicubic[1]:.6f} | "
    f"PSNR: {overall_bicubic[2]:.3f} dB | "
    f"Corr: {overall_bicubic[4]:.6f}"
)


# ============================================================
# NDVI - BICUBIC
# ============================================================

bicubic_ndvi = calculate_ndvi(
    bicubic[0],
    bicubic[3]
)


bicubic_ndvi_mae = mean_absolute_error(
    hr_ndvi.flatten(),
    bicubic_ndvi.flatten()
)

bicubic_ndvi_rmse = np.sqrt(
    mean_squared_error(
        hr_ndvi.flatten(),
        bicubic_ndvi.flatten()
    )
)

bicubic_ndvi_corr = np.corrcoef(
    hr_ndvi.flatten(),
    bicubic_ndvi.flatten()
)[0, 1]


print("\nBicubic NDVI:")
print(
    f"MAE: {bicubic_ndvi_mae:.6f} | "
    f"RMSE: {bicubic_ndvi_rmse:.6f} | "
    f"Corr: {bicubic_ndvi_corr:.6f}"
)


# ============================================================
# SEN2SR IMPROVEMENT OVER BICUBIC
# ============================================================

print("\n" + "=" * 40)
print("SEN2SR vs BICUBIC")
print("=" * 40)


print(
    f"Overall MAE improvement: "
    f"{overall_bicubic[0] - overall_sr[0]:.6f}"
)

print(
    f"Overall RMSE improvement: "
    f"{overall_bicubic[1] - overall_sr[1]:.6f}"
)

print(
    f"Overall PSNR improvement: "
    f"{overall_sr[2] - overall_bicubic[2]:.3f} dB"
)

print(
    f"Overall correlation improvement: "
    f"{overall_sr[4] - overall_bicubic[4]:.6f}"
)


# ============================================================
# PERCENTAGE IMPROVEMENT
# ============================================================

mae_improvement_percent = (
    (overall_bicubic[0] - overall_sr[0])
    / overall_bicubic[0]
) * 100


rmse_improvement_percent = (
    (overall_bicubic[1] - overall_sr[1])
    / overall_bicubic[1]
) * 100


print("\nPercentage improvement:")

print(
    f"MAE: {mae_improvement_percent:.2f}%"
)

print(
    f"RMSE: {rmse_improvement_percent:.2f}%"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 40)
print("FINAL SUMMARY")
print("=" * 40)


print(
    f"SEN2SR PSNR   : "
    f"{overall_sr[2]:.3f} dB"
)

print(
    f"Bicubic PSNR  : "
    f"{overall_bicubic[2]:.3f} dB"
)

print(
    f"SEN2SR MAE    : "
    f"{overall_sr[0]:.6f}"
)

print(
    f"Bicubic MAE   : "
    f"{overall_bicubic[0]:.6f}"
)

print(
    f"SEN2SR RMSE   : "
    f"{overall_sr[1]:.6f}"
)

print(
    f"Bicubic RMSE  : "
    f"{overall_bicubic[1]:.6f}"
)

print(
    f"SEN2SR Corr   : "
    f"{overall_sr[4]:.6f}"
)

print(
    f"Bicubic Corr  : "
    f"{overall_bicubic[4]:.6f}"
)

print("\nValidation completed successfully.")