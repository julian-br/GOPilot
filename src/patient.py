from dataclasses import dataclass


@dataclass(frozen=True)
class PreviousQuarterContact:
    quarter: str
    contact_type: str
    reason: str


@dataclass(frozen=True)
class Patient:
    id: str
    age: int
    gender: str
    insurance: str
    conditions: tuple[str, ...]
    billed_gops_current_quarter: tuple[str, ...]
    previous_quarter_contacts: tuple[PreviousQuarterContact, ...]

    @property
    def first_contact(self) -> bool:
        return not self.billed_gops_current_quarter
