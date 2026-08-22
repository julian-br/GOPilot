from langchain_core.documents import Document

from src.patient import Patient


def format_patient_context(patient: Patient | None) -> str:
    if patient is None:
        return "Nicht verfuegbar."
    previous_quarter_contacts = "\n".join(
        f"- {contact.quarter}: {contact.contact_type} wegen {contact.reason}"
        for contact in patient.previous_quarter_contacts
    ) or "keine dokumentiert"
    return (
        f"Alter: {patient.age}\n"
        f"Geschlecht: {patient.gender}\n"
        f"Versicherung: {patient.insurance}\n"
        f"Bekannte Diagnosen: {', '.join(patient.conditions) or 'keine'}\n"
        "Bereits in diesem Quartal abgerechnet: "
        f"{', '.join(patient.billed_gops_current_quarter) or 'keine'}\n"
        f"Erster Kontakt im Quartal: {'ja' if patient.first_contact else 'nein'}\n"
        f"Kontakte in vorherigen Quartalen:\n{previous_quarter_contacts}"
    )


def format_candidates(candidates: list[Document]) -> str:
    return "\n\n".join(
        f"{candidate.metadata['code']}\n{candidate.page_content}"
        for candidate in candidates
    )
