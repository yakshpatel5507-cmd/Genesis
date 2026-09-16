import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from scipy.ndimage import zoom
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr,
)

from opensr_degradation import pipe


# ============================================================
# CONFIGURATION
# ============================================================

ROIS = [
    "ROI_0000",
    "ROI_0001",
    "ROI_0002",
    "ROI_0003",
    "ROI_0004",
]

VALIDATION_DIR = Path("data/validation")
RESULTS_DIR = Path("results/multi_roi_validation")
MODEL_DIR = Path("models/SEN2SRLite_RGBN")

DEVICE = torch.device("cpu")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD SEN2SR MODEL
# ============================================================

print("=" * 60)
print("Loading SEN2SR model")
print("=" * 60)

# Make sure Python imports the correct load.py
sys.path.insert(
    0,
    str(MODEL_DIR.resolve())
)

from load import compiled_model


model = compiled_model(
    MODEL_DIR,
    DEVICE
)

print("SEN2SR model loaded.")
print("Device:", DEVICE)


# ============================================================
# OPEN-SR DEGRADATION
# ============================================================

degradation = pipe(
    "naip_d",
    add_noise=False,
    params={
        "reflectance_method": ["identity"]
    }
)

print("OpenSR degradation initialized.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_hr(path):

    with rasterio.open(path) as src:

        image = src.read().astype(
            np.float32
        )

        profile = src.profile.copy()

    # NAIP uint8 → [0, 1]
    image /= 255.0

    return image, profile


def degrade_hr(hr):

    """
    HR:
        (4, 484, 484)

    OpenSR:
        (1, 4, 121, 121)

    Return:
        (4, 121, 121)
    """

    tensor = torch.from_numpy(
        hr
    ).float()

   
    with torch.no_grad():

       lr, harmonized_hr = degradation.forward(tensor)

    # Remove method dimension
    lr = lr[0]

    return lr.cpu().numpy()


def run_sen2sr(lr):

    """
    LR:
        (4, 121, 121)

    SEN2SR requires:
        (1, 4, 128, 128)

    Output:
        (4, 484, 484)
    """

    channels, height, width = lr.shape

    target_size = 128

    pad_h = target_size - height
    pad_w = target_size - width

    lr_tensor = torch.from_numpy(
        lr
    ).float()

    # Reflect padding
    lr_tensor = F.pad(
        lr_tensor,
        (
            0,
            pad_w,
            0,
            pad_h
        ),
        mode="reflect"
    )

    # Add batch dimension
    lr_tensor = lr_tensor.unsqueeze(0)

    print(
        "  SEN2SR input:",
        tuple(lr_tensor.shape)
    )

    with torch.no_grad():

        sr = model(
            lr_tensor
        )

    print(
        "  SEN2SR raw output:",
        tuple(sr.shape)
    )

    sr = sr.squeeze(
        0
    ).cpu().numpy()

    # Remove padding
    sr = sr[
        :,
        :height * 4,
        :width * 4
    ]

    return sr


def bicubic_upscale(lr):

    """
    121x121 → 484x484
    """

    channels, h, w = lr.shape

    result = np.empty(
        (
            channels,
            h * 4,
            w * 4
        ),
        dtype=np.float32
    )

    for c in range(channels):

        result[c] = zoom(
            lr[c],
            4,
            order=3
        )

    return result


def calculate_metrics(
    prediction,
    reference
):

    prediction = np.asarray(
        prediction,
        dtype=np.float32
    )

    reference = np.asarray(
        reference,
        dtype=np.float32
    )

    mae = np.mean(
        np.abs(
            prediction - reference
        )
    )

    rmse = np.sqrt(
        np.mean(
            (prediction - reference) ** 2
        )
    )

    psnr_value = psnr(
        reference,
        prediction,
        data_range=1.0
    )

    correlation = np.corrcoef(
        prediction.ravel(),
        reference.ravel()
    )[0, 1]

    return (
        float(mae),
        float(rmse),
        float(psnr_value),
        float(correlation)
    )


def calculate_ndvi(image):

    """
    Band order:
        0 = Red
        1 = Green
        2 = Blue
        3 = NIR
    """

    red = image[0]
    nir = image[3]

    denominator = (
        nir + red + 1e-8
    )

    return (
        (nir - red)
        / denominator
    )


def ndvi_metrics(
    prediction,
    reference
):

    pred_ndvi = calculate_ndvi(
        prediction
    )

    ref_ndvi = calculate_ndvi(
        reference
    )

    mae = np.mean(
        np.abs(
            pred_ndvi - ref_ndvi
        )
    )

    rmse = np.sqrt(
        np.mean(
            (pred_ndvi - ref_ndvi) ** 2
        )
    )

    correlation = np.corrcoef(
        pred_ndvi.ravel(),
        ref_ndvi.ravel()
    )[0, 1]

    return (
        float(mae),
        float(rmse),
        float(correlation)
    )


def save_geotiff(
    path,
    data,
    profile
):

    out_profile = profile.copy()

    out_profile.update(
        {
            "height": data.shape[1],
            "width": data.shape[2],
            "count": data.shape[0],
            "dtype": "float32",
            "transform": rasterio.Affine(
                profile["transform"].a,
                profile["transform"].b,
                profile["transform"].c,
                profile["transform"].d,
                profile["transform"].e,
                profile["transform"].f,
            ),
        }
    )

    # HR transform is 2.5 m.
    # LR is only used for computation.
    # SR therefore uses the HR grid.

    with rasterio.open(
        path,
        "w",
        **out_profile
    ) as dst:

        dst.write(
            data.astype(
                np.float32
            )
        )


# ============================================================
# CSV STORAGE
# ============================================================

all_results = []


# ============================================================
# PROCESS EACH ROI
# ============================================================

for roi in ROIS:

    print("\n")
    print("=" * 60)
    print(f"PROCESSING {roi}")
    print("=" * 60)

    hr_path = (
        VALIDATION_DIR
        / roi
        / "hr.tif"
    )

    if not hr_path.exists():

        print(
            "ERROR: HR file missing:",
            hr_path
        )

        continue

    # --------------------------------------------------------
    # READ HR
    # --------------------------------------------------------

    hr, profile = read_hr(
        hr_path
    )

    print(
        "HR shape:",
        hr.shape
    )

    print(
        "HR CRS:",
        profile["crs"]
    )

    print(
        "HR resolution:",
        profile["transform"].a,
        abs(profile["transform"].e)
    )

    # --------------------------------------------------------
    # DEGRADE HR
    # --------------------------------------------------------

    print("\nCreating synthetic LR...")

    lr = degrade_hr(
        hr
    )

    print(
        "LR shape:",
        lr.shape
    )

    print(
        "LR range:",
        lr.min(),
        lr.max()
    )

    # --------------------------------------------------------
    # SEN2SR
    # --------------------------------------------------------

    print("\nRunning SEN2SR...")

    sr = run_sen2sr(
        lr
    )

    print(
        "SR shape:",
        sr.shape
    )

    # Clip for analysis
    sr = np.clip(
        sr,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # BICUBIC
    # --------------------------------------------------------

    print("\nCreating bicubic baseline...")

    bicubic = bicubic_upscale(
        lr
    )

    bicubic = np.clip(
        bicubic,
        0.0,
        1.0
    )

    print(
        "Bicubic shape:",
        bicubic.shape
    )

    # --------------------------------------------------------
    # SAVE SR
    # --------------------------------------------------------

    sr_path = (
        RESULTS_DIR
        / f"{roi}_SEN2SR.tif"
    )

    # HR already has 2.5m transform
    sr_profile = profile.copy()

    sr_profile.update(
        {
            "height": sr.shape[1],
            "width": sr.shape[2],
            "count": sr.shape[0],
            "dtype": "float32",
        }
    )

    with rasterio.open(
        sr_path,
        "w",
        **sr_profile
    ) as dst:

        dst.write(
            sr.astype(
                np.float32
            )
        )

    print(
        "Saved:",
        sr_path
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    sen2sr_metrics = calculate_metrics(
        sr,
        hr
    )

    bicubic_metrics = calculate_metrics(
        bicubic,
        hr
    )

    sr_ndvi = ndvi_metrics(
        sr,
        hr
    )

    bicubic_ndvi = ndvi_metrics(
        bicubic,
        hr
    )

    (
        sr_mae,
        sr_rmse,
        sr_psnr,
        sr_corr
    ) = sen2sr_metrics

    (
        bi_mae,
        bi_rmse,
        bi_psnr,
        bi_corr
    ) = bicubic_metrics

    (
        sr_ndvi_mae,
        sr_ndvi_rmse,
        sr_ndvi_corr
    ) = sr_ndvi

    (
        bi_ndvi_mae,
        bi_ndvi_rmse,
        bi_ndvi_corr
    ) = bicubic_ndvi

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nResults:")
    print("-" * 60)

    print(
        f"SEN2SR  MAE : {sr_mae:.6f}"
    )

    print(
        f"Bicubic MAE : {bi_mae:.6f}"
    )

    print(
        f"SEN2SR  RMSE: {sr_rmse:.6f}"
    )

    print(
        f"Bicubic RMSE: {bi_rmse:.6f}"
    )

    print(
        f"SEN2SR  PSNR: {sr_psnr:.3f} dB"
    )

    print(
        f"Bicubic PSNR: {bi_psnr:.3f} dB"
    )

    print(
        f"SEN2SR  Corr: {sr_corr:.6f}"
    )

    print(
        f"Bicubic Corr: {bi_corr:.6f}"
    )

    print(
        f"SEN2SR NDVI MAE: {sr_ndvi_mae:.6f}"
    )

    print(
        f"Bicubic NDVI MAE: {bi_ndvi_mae:.6f}"
    )

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    all_results.append(
        {
            "ROI": roi,

            "SEN2SR_MAE": sr_mae,
            "SEN2SR_RMSE": sr_rmse,
            "SEN2SR_PSNR": sr_psnr,
            "SEN2SR_Correlation": sr_corr,

            "Bicubic_MAE": bi_mae,
            "Bicubic_RMSE": bi_rmse,
            "Bicubic_PSNR": bi_psnr,
            "Bicubic_Correlation": bi_corr,

            "SEN2SR_NDVI_MAE": sr_ndvi_mae,
            "SEN2SR_NDVI_RMSE": sr_ndvi_rmse,
            "SEN2SR_NDVI_Correlation": sr_ndvi_corr,

            "Bicubic_NDVI_MAE": bi_ndvi_mae,
            "Bicubic_NDVI_RMSE": bi_ndvi_rmse,
            "Bicubic_NDVI_Correlation": bi_ndvi_corr,
        }
    )


# ============================================================
# SAVE CSV
# ============================================================

import csv

csv_path = (
    RESULTS_DIR
    / "all_roi_results.csv"
)

if all_results:

    keys = all_results[0].keys()

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=keys
        )

        writer.writeheader()

        writer.writerows(
            all_results
        )

    print(
        "\nSaved results:",
        csv_path
    )


# ============================================================
# MEAN ± STD
# ============================================================

if all_results:

    print("\n")
    print("=" * 70)
    print("FINAL MULTI-ROI RESULTS")
    print("=" * 70)

    metric_pairs = [
        (
            "MAE",
            "SEN2SR_MAE",
            "Bicubic_MAE"
        ),
        (
            "RMSE",
            "SEN2SR_RMSE",
            "Bicubic_RMSE"
        ),
        (
            "PSNR",
            "SEN2SR_PSNR",
            "Bicubic_PSNR"
        ),
        (
            "Correlation",
            "SEN2SR_Correlation",
            "Bicubic_Correlation"
        ),
    ]

    for name, sr_key, bi_key in metric_pairs:

        sr_values = np.array(
            [
                r[sr_key]
                for r in all_results
            ]
        )

        bi_values = np.array(
            [
                r[bi_key]
                for r in all_results
            ]
        )

        print(
            f"\n{name}:"
        )

        print(
            f"  SEN2SR : "
            f"{sr_values.mean():.6f} "
            f"± {sr_values.std():.6f}"
        )

        print(
            f"  Bicubic: "
            f"{bi_values.mean():.6f} "
            f"± {bi_values.std():.6f}"
        )

    # NDVI
    for name, sr_key, bi_key in [
        (
            "NDVI MAE",
            "SEN2SR_NDVI_MAE",
            "Bicubic_NDVI_MAE"
        ),
        (
            "NDVI RMSE",
            "SEN2SR_NDVI_RMSE",
            "Bicubic_NDVI_RMSE"
        ),
        (
            "NDVI Correlation",
            "SEN2SR_NDVI_Correlation",
            "Bicubic_NDVI_Correlation"
        ),
    ]:

        sr_values = np.array(
            [
                r[sr_key]
                for r in all_results
            ]
        )

        bi_values = np.array(
            [
                r[bi_key]
                for r in all_results
            ]
        )

        print(
            f"\n{name}:"
        )

        print(
            f"  SEN2SR : "
            f"{sr_values.mean():.6f} "
            f"± {sr_values.std():.6f}"
        )

        print(
            f"  Bicubic: "
            f"{bi_values.mean():.6f} "
            f"± {bi_values.std():.6f}"
        )

else:

    print(
        "\nNo ROIs were successfully processed."
    )


print("\n")
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)