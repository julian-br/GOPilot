"""Read the KBV EBM catalogue (SDEBM record type 850).

Every element carries its value either in the V attribute or as text, never both.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

from src.ebm.gop import GOP
from src.paths import EBM_MASTER

NAMESPACE = {"go": "urn:ehd/go/001"}
GOP_TAG = "{urn:ehd/go/001}gnr"


def load_gops(path: Path = EBM_MASTER) -> list[GOP]:
    root = ET.parse(path).getroot()
    return [_to_gop(element) for element in root.iter(GOP_TAG) if _is_wanted(element)]


def _is_wanted(element: ET.Element) -> bool:
    """Definitions carry VT, cross-references carry DN. Codes with a letter suffix are
    regional variants we skip for now."""
    return "VT" in element.attrib and element.get("V").isdigit()


def _to_gop(element: ET.Element) -> GOP:
    specialties = element.findall(
        "go:bedingung/go:fachgruppe_liste/go:versorgungsbereich/go:fachgruppe", NAMESPACE)
    return GOP(
        code=element.get("V"),
        short_text=_value(element, "go:allgemein/go:legende/go:kurztext"),
        long_text=_value(element, "go:allgemein/go:legende/go:langtext"),
        obligatory_content=_value(element, "go:allgemein/go:leistungsinhalt_obligat"),
        specialties=tuple(s.get("V") for s in specialties),
    )


def _value(element: ET.Element, path: str) -> str:
    found = element.find(path, NAMESPACE)
    if found is None:
        return ""
    return " ".join((found.get("V") or "".join(found.itertext())).split())
