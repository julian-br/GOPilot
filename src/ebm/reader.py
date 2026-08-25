"""Read the KBV EBM catalogue (SDEBM record type 850).

Every element carries its value either in the V attribute or as text, never both.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

from src.ebm.gop import AgeLimit, GOP, OccurrenceLimit
from src.ebm.keytables import KeyLabels, load_key_labels
from src.paths import EBM_KEYTABS, EBM_MASTER

NAMESPACE = {"go": "urn:ehd/go/001"}
GOP_TAG = "{urn:ehd/go/001}gnr"
HEADER_TAG = "{urn:ehd/001}header"
SERVICE_PERIOD_TAG = "{urn:ehd/001}service_tmr"


def load_gops(
    path: Path = EBM_MASTER,
    keytabs_path: Path = EBM_KEYTABS,
) -> list[GOP]:
    key_labels = load_key_labels(keytabs_path)
    root = ET.parse(path).getroot()
    return [
        _to_gop(element, key_labels)
        for element in root.iter(GOP_TAG)
        if _is_wanted(element)
    ]


def load_quarter(path: Path = EBM_MASTER) -> str:
    root = ET.parse(path).getroot()
    period = root.find(f"{HEADER_TAG}/{SERVICE_PERIOD_TAG}")
    if period is None or not period.get("V"):
        raise ValueError("EBM catalogue has no service period")
    year, month, _ = map(int, period.get("V").split("..", 1)[0].split("-"))
    return f"{(month - 1) // 3 + 1}/{year}"


def _is_wanted(element: ET.Element) -> bool:
    """Definitions carry VT, cross-references carry DN. Codes with a letter suffix are
    regional variants we skip for now."""
    return "VT" in element.attrib and element.get("V").isdigit()


def _to_gop(element: ET.Element, key_labels: KeyLabels) -> GOP:
    specialties = element.findall(
        "go:bedingung/go:fachgruppe_liste/go:versorgungsbereich/go:fachgruppe", NAMESPACE)
    return GOP(
        code=element.get("V"),
        short_text=_value(element, "go:allgemein/go:legende/go:kurztext"),
        long_text=_value(element, "go:allgemein/go:legende/go:langtext"),
        obligatory_content=_value(element, "go:allgemein/go:leistungsinhalt_obligat"),
        annotations=tuple(
            _values(element, "go:allgemein/go:anmerkungen_liste/go:anmerkung")
        ),
        billing_text=_value(element, "go:bedingung/go:abr_best"),
        occurrence_limits=_occurrence_limits(element, key_labels),
        code_type=_value(element, "go:kv/go:kennzeichen/go:gnr_type_cd"),
        specialties=tuple(s.get("V") for s in specialties),
    )


def _value(element: ET.Element, path: str) -> str:
    found = element.find(path, NAMESPACE)
    if found is None:
        return ""
    return " ".join((found.get("V") or "".join(found.itertext())).split())


def _values(element: ET.Element, path: str) -> list[str]:
    return [
        value
        for found in element.findall(path, NAMESPACE)
        if (value := " ".join((found.get("V") or "".join(found.itertext())).split()))
    ]


def _occurrence_limits(
    element: ET.Element,
    key_labels: KeyLabels,
) -> tuple[OccurrenceLimit, ...]:
    limits: list[OccurrenceLimit] = []
    for scope in element.findall(
        "go:bedingung/go:anzahlbedingung_liste/go:bezugsraum",
        NAMESPACE,
    ):
        count = scope.find("go:anzahl", NAMESPACE)
        count_value = count.get("V") if count is not None else None
        scope_domain = scope.get("U-DOMAIN")
        scope_code = scope.get("U")
        if count_value is None or not scope_domain or not scope_code:
            continue
        limits.append(
            OccurrenceLimit(
                max_occurrences=int(count_value),
                reference_scope=key_labels.get((scope_domain, scope_code)),
                reference_scope_code=scope_code,
                exception_codes=tuple(
                    item.get("V")
                    for item in scope.findall(
                        "go:aussetzungsgrund_liste/go:gnr_zusatzangabe",
                        NAMESPACE,
                    )
                    if item.get("V")
                ),
                age_limits=_age_limits(scope, key_labels),
            )
        )
    return tuple(limits)


def _age_limits(
    element: ET.Element,
    key_labels: KeyLabels,
) -> tuple[AgeLimit, ...]:
    limits: list[AgeLimit] = []
    for age in element.findall("go:altersbedingung_liste/go:alter", NAMESPACE):
        boundary = age.find("go:range_typ", NAMESPACE)
        boundary_value = boundary.get("V") if boundary is not None else None
        age_value = age.get("V")
        unit_domain = age.get("U-DOMAIN")
        unit_code = age.get("U")
        if not boundary_value or age_value is None or not unit_domain or not unit_code:
            continue
        limits.append(
            AgeLimit(
                boundary=boundary_value,
                value=int(age_value),
                unit=key_labels.get((unit_domain, unit_code)),
                unit_code=unit_code,
            )
        )
    return tuple(limits)
