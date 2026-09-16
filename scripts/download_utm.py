import requests
import getpass
from pathlib import Path
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session


# --------------------------------------------------
# 1. Copernicus authentication
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

PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"


# --------------------------------------------------
# 3. Ahmedabad-area UTM tile
# --------------------------------------------------
#
# UTM Zone 43N = EPSG:32643
#
# We'll first transform our approximate Ahmedabad
# coordinates into UTM.
#
# This is approximately:
#
# Longitude 72.50 E
# Latitude  23.00 N
#
# → Easting  ~243,000 m
# → Northing ~2,545,000 m
#
# We request a 1280 m × 1280 m tile:
#
# 128 pixels × 10 m = 1280 m
#

bbox = [
    242500,
    2544500,
    243780,
    2545780
]


# --------------------------------------------------
# 4. Evalscript
# --------------------------------------------------

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


# --------------------------------------------------
# 5. Request body
# --------------------------------------------------

request_body = {

    "input": {

        "bounds": {

            "bbox": bbox,

            "properties": {
                "crs":
                "http://www.opengis.net/def/crs/EPSG/0/32643"
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
# 6. Send request
# --------------------------------------------------

headers = {

    "Authorization":
    f"Bearer {access_token}",

    "Content-Type":
    "application/json"
}


print("Requesting UTM Sentinel-2 tile...")

response = requests.post(
    PROCESS_URL,
    headers=headers,
    json=request_body
)


# --------------------------------------------------
# 7. Error handling
# --------------------------------------------------

if response.status_code != 200:

    print("\nDownload failed.")

    print("Status:", response.status_code)

    print(response.text)

    raise SystemExit(1)


# --------------------------------------------------
# 8. Save
# --------------------------------------------------

output_dir = Path("data/raw")

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


output_file = (
    output_dir /
    "sentinel2_utm_128_B02_B03_B04_B08.tif"
)


with open(output_file, "wb") as f:

    f.write(response.content)


print("\nDownload successful!")

print("Saved:", output_file)

print(
    "Size:",
    round(
        output_file.stat().st_size / 1024,
        2
    ),
    "KB"
)