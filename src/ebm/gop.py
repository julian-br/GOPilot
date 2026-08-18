from dataclasses import dataclass


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
    specialties: tuple[str, ...]

    def billable_by(self, specialty: str) -> bool:
        return not self.specialties or specialty in self.specialties

    @property
    def embedding_text(self) -> str:
        parts = (self.short_text, self.long_text, self.obligatory_content)
        return "\n".join(dict.fromkeys(part for part in parts if part))
