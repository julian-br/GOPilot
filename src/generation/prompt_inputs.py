import json

from langchain_core.documents import Document

from src.patient import Patient, PriorContact


def format_patient_context(
    patient: Patient | None, current_quarter: str
) -> str:
    if patient is None:
        return "Keine Patientendaten oder vergangenen Kontakte dokumentiert."

    def format_contact(contact: PriorContact) -> str:
        return (
            f"- {contact.quarter}: {contact.contact_type} wegen {contact.reason}; "
            f"abgerechnet: {', '.join(contact.billed_gops) or 'keine'}"
        )

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
        f"Vergangene Kontakte im Abrechnungsquartal {current_quarter} vor dem aktuellen, "
        "abzurechnenden Besuch:\n"
        f"{current_context or 'keine dokumentiert'}\n"
        "Vergangene Kontakte aus frueheren Quartalen (kein aktueller Besuch):\n"
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
    return "\n\n".join(format_candidate(candidate) for candidate in candidates)


def format_candidate(candidate: Document) -> str:
    parts = [f"GOP {candidate.metadata['code']}", candidate.page_content]
    annotations = candidate.metadata.get("annotations")
    if annotations:
        parts.append(
            "Anmerkungen:\n"
            + json.dumps(annotations, ensure_ascii=False, indent=2)
        )
    billing_rules = candidate.metadata.get("billing_rules")
    if billing_rules:
        parts.append(
            "Abrechnungsregeln:\n"
            + json.dumps(billing_rules, ensure_ascii=False, indent=2)
        )
    return "\n".join(parts)
