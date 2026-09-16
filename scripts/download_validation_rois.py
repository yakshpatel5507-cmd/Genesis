import os
import requests


BASE_URL = (
    "https://huggingface.co/datasets/"
    "isp-uv-es/SEN2NAIP/resolve/main/"
    "demo/cross-sensor"
)

OUTPUT_DIR = "data/validation"

# ROI_0000 already exists, so download 0001-0004
ROIS = [
    "ROI_0001",
    "ROI_0002",
    "ROI_0003",
    "ROI_0004",
]


def download_file(url, output_path):

    print(f"\nDownloading:")
    print(url)

    response = requests.get(
        url,
        stream=True,
        timeout=120
    )

    response.raise_for_status()

    total = int(
        response.headers.get(
            "content-length",
            0
        )
    )

    downloaded = 0

    with open(output_path, "wb") as f:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if chunk:

                f.write(chunk)

                downloaded += len(chunk)

                if total:
                    percent = (
                        downloaded / total
                    ) * 100

                    print(
                        f"\rProgress: {percent:.1f}%",
                        end=""
                    )

    print("\nDone.")


for roi in ROIS:

    roi_dir = os.path.join(
        OUTPUT_DIR,
        roi
    )

    os.makedirs(
        roi_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        roi_dir,
        "hr.tif"
    )

    url = (
        f"{BASE_URL}/"
        f"{roi}/hr.tif"
    )

    try:

        download_file(
            url,
            output_path
        )

        print(
            f"Saved: {output_path}"
        )

    except Exception as e:

        print(
            f"FAILED {roi}: {e}"
        )


print("\nDownload process finished.")