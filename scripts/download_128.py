import requests
import json
import getpass
from pathlib import Path
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

# --------------------------------------------------
# 1. Copernicus credentials
# --------------------------------------------------

CLIENT_ID = input("Enter Copernicus Client ID: ")
CLIENT_SECRET = getpass.getpass("Enter Copernicus Client Secret: ")

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

client = BackendApplicationClient(client_id=CLIENT_ID)
oauth = OAuth2Session(client=client)

token = oauth.fetch_token(
    token_url=TOKEN_URL,
    client_secret=CLIENT_SECRET,
    include_client_id=True
)

access_token = token["access_token"]

print("\nAuthentication successful!")


# --------------------------------------------------
# 2. Sentinel Hub Process API
# --------------------------------------------------

PROCESS_URL = (
    "https://sh.dataspace.copernicus.eu/api/v1/process"
)

# Ahmedabad-area bounding box
#
# We use EPSG:4326 for this first model test.
# The exact geospatial projection will be improved
# later for the final prototype.

bbox = [72.50, 23.00, 72.5128, 23.0128]


evalscript = """
//VERSION=3

function setup() {
    return {
        input: [{
            bands: ["B02", "B03", "B04", "B08"],
            units: "REFLECTANCE"
        }],
        output: {
            bands: 4,
            sampleType: "FLOAT32"
        }
    };
}

function evaluatePixel(sample) {
    return [
        sample.B02,
        sample.B03,
        sample.B04,
        sample.B08
    ];
}
"""

request_body = {
    "input": {
        "bounds": {
            "bbox": bbox,
            "properties": {
                "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
            }
        },
        "data": [{
            "type": "sentinel-2-l2a",
            "dataFilter": {
                "timeRange": {
                    "from": "2025-12-27T00:00:00Z",
                    "to": "2025-12-28T00:00:00Z"
                },
                "maxCloudCoverage": 10
            }
        }]
    },

    "output": {
        "width": 128,
        "height": 128,
        "responses": [{
            "identifier": "default",
            "format": {
                "type": "image/tiff"
            }
        }]
    },

    "evalscript": evalscript
}


# --------------------------------------------------
# 3. Request image
# --------------------------------------------------

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

print("Requesting Sentinel-2 tile...")

response = requests.post(
    PROCESS_URL,
    headers=headers,
    json=request_body
)

if response.status_code != 200:
    print("\nDownload failed.")
    print("Status:", response.status_code)
    print(response.text)
    raise SystemExit(1)


# --------------------------------------------------
# 4. Save image
# --------------------------------------------------

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = (
    output_dir /
    "sentinel2_128_B02_B03_B04_B08.tif"
)

with open(output_file, "wb") as f:
    f.write(response.content)

print("\nDownload successful!")
print("Saved:", output_file)
print("Size:", round(output_file.stat().st_size / 1024, 2), "KB")