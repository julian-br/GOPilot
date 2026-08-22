"""Validation and conversion for EBM catalogue versions."""

import re

QUARTER_PATTERN = re.compile(r"^[1-4]/\d{4}$")
COLLECTION_PATTERN = re.compile(r"^ebm_(\d{4})_q([1-4])(?:_|$)")


def parse_quarter(value: str) -> str:
    if not QUARTER_PATTERN.fullmatch(value):
        raise ValueError(f"invalid EBM quarter {value!r}; expected <1-4>/<year>")
    return value


def collection_quarter(collection_name: str) -> str:
    match = COLLECTION_PATTERN.match(collection_name)
    if match is None:
        raise ValueError(
            f"invalid EBM collection {collection_name!r}; expected ebm_<year>_q<1-4>"
        )
    year, quarter = match.groups()
    return f"{quarter}/{year}"
