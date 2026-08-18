from dataclasses import dataclass


@dataclass(frozen=True)
class GOP:
    """One billing code of the EBM catalogue.

    An empty `care_areas` means the code carries no specialty restriction, so it is
    billable everywhere — not that it is billable nowhere.
    """

    code: str
    short_text: str
    long_text: str
    obligatory_content: str
    care_areas: tuple[str, ...]

    def billable_in(self, care_area: str) -> bool:
        return not self.care_areas or care_area in self.care_areas

    @property
    def embedding_text(self) -> str:
        parts = (self.short_text, self.long_text, self.obligatory_content)
        return "\n".join(dict.fromkeys(part for part in parts if part))
