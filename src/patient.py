from dataclasses import dataclass


@dataclass(frozen=True)
class PriorContact:
    quarter: str
    contact_type: str
    reason: str
    billed_gops: tuple[str, ...]


@dataclass(frozen=True)
class Patient:
    id: str
    age: int
    gender: str
    insurance: str
    conditions: tuple[str, ...]
    prior_contacts: tuple[PriorContact, ...]
