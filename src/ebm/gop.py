from dataclasses import dataclass


@dataclass(frozen=True)
class AgeLimit:
    boundary: str
    value: int
    unit: str | None
    unit_code: str


@dataclass(frozen=True)
class OccurrenceLimit:
    max_occurrences: int
    reference_scope: str | None
    reference_scope_code: str
    exception_codes: tuple[str, ...]
    age_limits: tuple[AgeLimit, ...]


@dataclass(frozen=True)
class GOP:
    """One billing code of the EBM catalogue.

    An empty `specialties` means the code carries no restriction, so any practice may bill
    it — not that nobody may.
    """

    code: str
    short_text: str
    long_text: str
    obligatory_content: str
    annotations: tuple[str, ...]
    billing_text: str
    occurrence_limits: tuple[OccurrenceLimit, ...]
    code_type: str
    specialties: tuple[str, ...]

    def billable_by(self, specialty: str) -> bool:
        return not self.specialties or specialty in self.specialties

    @property
    def embedding_text(self) -> str:
        parts = (self.short_text, self.long_text, self.obligatory_content)
        return "\n".join(dict.fromkeys(part for part in parts if part))
