import requests
import zipfile
import struct
import os


ZIP_URL = (
    "https://huggingface.co/datasets/"
    "isp-uv-es/SEN2NAIP/resolve/main/"
    "cross-sensor/cross-sensor.zip"
)

ROIS = [
    "ROI_0000",
    "ROI_0001",
    "ROI_0002",
    "ROI_0003",
    "ROI_0004",
]

OUTPUT_DIR = "data/validation"


def get_zip_info():

    print("Reading ZIP information...")

    r = requests.head(
        ZIP_URL,
        allow_redirects=True
    )

    r.raise_for_status()

    size = int(
        r.headers["Content-Length"]
    )

    # Download final 10 MB
    tail_size = 10_000_000
    start = max(
        0,
        size - tail_size
    )

    r = requests.get(
        ZIP_URL,
        headers={
            "Range":
            f"bytes={start}-{size - 1}"
        },
        allow_redirects=True,
        timeout=120
    )

    r.raise_for_status()

    tail = r.content

    # Find End Of Central Directory
    eocd_sig = b"PK\x05\x06"

    pos = tail.rfind(eocd_sig)

    if pos == -1:
        raise RuntimeError(
            "ZIP End Of Central Directory not found."
        )

    eocd = tail[pos:pos + 22]

    (
        sig,
        disk,
        cd_disk,
        entries_disk,
        entries_total,
        cd_size,
        cd_offset,
        comment_length,
    ) = struct.unpack(
        "<4sHHHHIIH",
        eocd
    )

    print(
        f"ZIP entries: {entries_total}"
    )

    # Extract central directory
    cd_start = cd_offset - start

    central = tail[
        cd_start:
        cd_start + cd_size
    ]

    files = {}

    p = 0

    while p < len(central):

        if central[p:p + 4] != b"PK\x01\x02":
            break

        header = central[p:p + 46]

        (
            sig,
            version_made,
            version_needed,
            flags,
            compression,
            mod_time,
            mod_date,
            crc,
            compressed_size,
            uncompressed_size,
            filename_length,
            extra_length,
            comment_length,
            disk_number,
            internal_attr,
            external_attr,
            local_offset,
        ) = struct.unpack(
            "<4sHHHHHHIIIHHHHHII",
            header
        )

        filename_start = p + 46

        filename = central[
            filename_start:
            filename_start + filename_length
        ].decode(
            "utf-8"
        )

        files[filename] = {
            "compression": compression,
            "compressed_size": compressed_size,
            "uncompressed_size": uncompressed_size,
            "local_offset": local_offset,
        }

        p += (
            46
            + filename_length
            + extra_length
            + comment_length
        )

    return files


def download_member(
    member_name,
    info,
    output_path
):

    print(
        f"\nDownloading {member_name}"
    )

    local_offset = info[
        "local_offset"
    ]

    # Read local file header first.
    r = requests.get(
        ZIP_URL,
        headers={
            "Range":
            f"bytes={local_offset}-{local_offset + 29}"
        },
        allow_redirects=True,
        timeout=120
    )

    r.raise_for_status()

    header = r.content

    if header[:4] != b"PK\x03\x04":

        raise RuntimeError(
            "Invalid ZIP local header."
        )

    filename_length = struct.unpack(
        "<H",
        header[26:28]
    )[0]

    extra_length = struct.unpack(
        "<H",
        header[28:30]
    )[0]

    data_start = (
        local_offset
        + 30
        + filename_length
        + extra_length
    )

    data_end = (
        data_start
        + info["compressed_size"]
        - 1
    )

    print(
        f"Compressed size: "
        f"{info['compressed_size'] / 1024**2:.2f} MB"
    )

    print(
        f"Downloading bytes "
        f"{data_start:,} - {data_end:,}"
    )

    r = requests.get(
        ZIP_URL,
        headers={
            "Range":
            f"bytes={data_start}-{data_end}"
        },
        allow_redirects=True,
        timeout=300
    )

    r.raise_for_status()

    compressed_data = r.content

    print(
        f"Received: "
        f"{len(compressed_data) / 1024**2:.2f} MB"
    )

    # Decompress according to ZIP compression method.
    if info["compression"] == 0:

        data = compressed_data

    elif info["compression"] == 8:

        import zlib

        decompressor = zlib.decompressobj(
            -15
        )

        data = (
            decompressor.decompress(
                compressed_data
            )
            + decompressor.flush()
        )

    else:

        raise RuntimeError(
            "Unsupported ZIP compression method: "
            + str(info["compression"])
        )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    with open(
        output_path,
        "wb"
    ) as f:

        f.write(data)

    print(
        f"Saved: {output_path}"
    )


files = get_zip_info()

print(
    f"\nFound {len(files)} files in ZIP."
)


for roi in ROIS:

    for filename in [
        f"cross-sensor/{roi}/hr.tif",
        f"cross-sensor/{roi}/metadata.json",
    ]:

        if filename not in files:

            print(
                f"NOT FOUND: {filename}"
            )

            continue

        relative = filename.replace(
            "cross-sensor/",
            ""
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            relative
        )

        # Skip existing files
        if os.path.exists(
            output_path
        ):

            print(
                f"Already exists: "
                f"{output_path}"
            )

            continue

        download_member(
            filename,
            files[filename],
            output_path
        )


print(
    "\n================================"
)

print(
    "Selected ROI download complete."
)

print(
    "================================"
)