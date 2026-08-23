from langchain_core.documents import Document

from src.patient import Patient, PriorContact


def format_patient_context(
    patient: Patient | None, current_quarter: str | None = None
) -> str:
    if patient is None:
        return "Nicht verfuegbar."

    def format_contact(contact: PriorContact) -> str:
        return (
            f"- {contact.quarter}: {contact.contact_type} wegen {contact.reason}; "
            f"abgerechnet: {', '.join(contact.billed_gops) or 'keine'}"
        )

    if current_quarter is None:
        contact_context = "Bisherige Kontakte vor diesem Fall:\n" + (
            "\n".join(format_contact(contact) for contact in patient.prior_contacts)
            or "keine dokumentiert"
        )
    else:
        current_contacts = [
            contact
            for contact in patient.prior_contacts
            if contact.quarter == current_quarter
        ]
        historical_contacts = [
            contact
            for contact in patient.prior_contacts
            if contact.quarter != current_quarter
        ]
        current_context = "\n".join(format_contact(contact) for contact in current_contacts)
        historical_context = "\n".join(
            format_contact(contact) for contact in historical_contacts
        )
        contact_context = (
            f"Kontakte im Abrechnungsquartal {current_quarter} vor diesem Fall:\n"
            f"{current_context or 'keine dokumentiert'}\n"
            "Fruehere Kontakte (deren GOPs sind kein Abrechnungsausschluss im aktuellen Quartal):\n"
            f"{historical_context or 'keine dokumentiert'}"
        )

    return (
        f"Alter: {patient.age}\n"
        f"Geschlecht: {patient.gender}\n"
        f"Versicherung: {patient.insurance}\n"
        f"Bekannte Diagnosen: {', '.join(patient.conditions) or 'keine'}\n"
        f"{contact_context}"
    )


def format_candidates(candidates: list[Document]) -> str:
    return "\n\n".join(
        f"{candidate.metadata['code']}\n{candidate.page_content}"
        for candidate in candidates
    )
