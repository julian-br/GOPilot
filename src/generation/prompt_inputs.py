from src.db.patients import Patient


def format_patient_context(patient: Patient | None) -> str:
    if patient is None:
        return "Nicht verfuegbar."
    return "\n".join(
        [
            f"Alter: {patient.age}",
            f"Geschlecht: {patient.gender}",
            f"Versicherung: {patient.insurance}",
            f"Bekannte Diagnosen: {', '.join(patient.conditions) or 'keine'}",
            f"Bereits in diesem Quartal abgerechnet: {', '.join(patient.billed_gops) or 'keine'}",
            f"Erster Kontakt im Quartal: {'ja' if patient.first_contact else 'nein'}",
        ]
    )
