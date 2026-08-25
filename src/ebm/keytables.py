"""Resolve coded SDEBM values through the key tables shipped by the KBV."""

import xml.etree.ElementTree as ET
from pathlib import Path

from src.paths import EBM_KEYTABS

KEY_TAG = "{urn:ehd/001}key"
KeyLabels = dict[tuple[str, str], str]


def load_key_labels(path: Path = EBM_KEYTABS) -> KeyLabels:
    """Return human-readable labels keyed by ``(domain OID, value)``.

    One KBV package contains one applicable version per domain. Conflicting labels
    therefore indicate that incompatible key-table versions were mixed locally.
    """
    labels: KeyLabels = {}
    for source in sorted(path.glob("*.xml")):
        for key in ET.parse(source).getroot().iter(KEY_TAG):
            domain = key.get("S")
            value = key.get("V")
            label = key.get("DN")
            if not domain or not value or not label:
                continue
            lookup = (domain, value)
            normalized = " ".join(label.split())
            previous = labels.get(lookup)
            if previous is not None and previous != normalized:
                raise ValueError(
                    f"conflicting labels for key-table value {domain}/{value}"
                )
            labels[lookup] = normalized
    if not labels:
        raise FileNotFoundError(f"no KBV key tables found in {path}")
    return labels
