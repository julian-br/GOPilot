"""Download the KBV EBM catalogue (SDEBM master data, record type 850)."""

import io
import re
import struct
import urllib.request
import zipfile
from pathlib import PurePosixPath

from src.paths import EBM, EBM_KEYTABS, EBM_MASTER

INDEX_URL = "https://update.kbv.de/ita-update/Stammdateien/KBV_Stammdateien/"
PAYLOAD = "resources/packs/pack-KBV-Stammdateien"


def main() -> None:
    payload = zipfile.ZipFile(io.BytesIO(_read(_jar_url()))).read(PAYLOAD)
    archive = _largest_zip(payload)
    master_files = [
        name
        for name in archive.namelist()
        if PurePosixPath(name).name.startswith("850") and name.endswith(".xml")
    ]
    keytab_files = [
        name
        for name in archive.namelist()
        if PurePosixPath(name).name.startswith("S_") and name.endswith(".xml")
    ]
    if not master_files:
        raise FileNotFoundError("record type 850 is missing from the KBV archive")
    if not keytab_files:
        raise FileNotFoundError("key tables are missing from the KBV archive")

    EBM.mkdir(parents=True, exist_ok=True)
    EBM_KEYTABS.mkdir(parents=True, exist_ok=True)
    EBM_MASTER.write_bytes(archive.read(sorted(master_files)[0]))
    print(f"  {EBM_MASTER}")
    for previous in EBM_KEYTABS.glob("*.xml"):
        previous.unlink()
    for name in keytab_files:
        filename = PurePosixPath(name).name
        (EBM_KEYTABS / filename).write_bytes(archive.read(name))
    print(f"  {len(keytab_files)} key tables in {EBM_KEYTABS}")


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
