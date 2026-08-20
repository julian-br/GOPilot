from langchain_core.prompts import ChatPromptTemplate

BILLING_WITHOUT_RETRIEVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Du empfiehlst EBM-GOP-Abrechnungsziffern fuer eine deutsche Arztpraxis. "
            "Gib nur fuenfstellige GOP-Ziffern zurueck, die durch Diktat und Patientenkontext begruendet sind. "
            "Denke auch an Pauschalen, wenn Patientenkontakt, Quartalsstatus und Kontext dafuer sprechen. "
            "Erfinde keine Leistungen, Diagnosen oder Befunde. Wenn du unsicher bist, lasse die Ziffer weg. "
            "Begruende jede Ziffer kurz.",
        ),
        (
            "human",
            "Diktat:\n{dictation}\n\nPatientenkontext:\n{patient_context}",
        ),
    ]
)
