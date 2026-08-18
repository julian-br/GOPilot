"""Download the KBV EBM catalogue (SDEBM master data, record type 850)."""

import io
import re
import struct
import urllib.request
import zipfile

from src.paths import EBM, EBM_MASTER

INDEX_URL = "https://update.kbv.de/ita-update/Stammdateien/KBV_Stammdateien/"
PAYLOAD = "resources/packs/pack-KBV-Stammdateien"


def main() -> None:
    EBM.mkdir(parents=True, exist_ok=True)
    payload = zipfile.ZipFile(io.BytesIO(_read(_jar_url()))).read(PAYLOAD)
    archive = _largest_zip(payload)
    for name in archive.namelist():
        if name.startswith("850") and name.endswith(".xml"):
            EBM_MASTER.write_bytes(archive.read(name))
            print(f"  {EBM_MASTER}")


def _jar_url() -> str:
    listing = _read(INDEX_URL).decode("utf-8", "replace")
    return INDEX_URL + sorted(re.findall(r'href="(kbv_stammdateien[^"]+\.jar)"', listing))[-1]


def _read(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=600) as response:
        return response.read()


def _largest_zip(payload: bytes) -> zipfile.ZipFile:
    found = (_zip_at(payload, m.start()) for m in re.finditer(rb"PK\x05\x06", payload))
    return max((z for z in found if z), key=lambda z: sum(i.file_size for i in z.infolist()))


def _zip_at(payload: bytes, eocd: int) -> zipfile.ZipFile | None:
    size, offset, comment = struct.unpack("<IIH", payload[eocd + 12:eocd + 22])
    start = eocd - size - offset
    if start < 0:
        return None
    try:
        return zipfile.ZipFile(io.BytesIO(payload[start:eocd + 22 + comment]))
    except zipfile.BadZipFile:
        return None


if __name__ == "__main__":
    main()
