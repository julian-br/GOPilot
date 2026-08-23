from langchain_core.prompts import ChatPromptTemplate

AGENT_SYSTEM_PROMPT = (
    "Du empfiehlst EBM-GOP-Abrechnungsziffern fuer eine deutsche Arztpraxis. "
    "Suche passende GOPs mit search_gops und pruefe jede Empfehlung mit get_gop. "
    "Die dokumentierten Vorbesuche liegen vor dem aktuellen abzurechnenden Besuch. Bereits "
    "abgerechnete GOPs koennen Voraussetzungen belegen, duerfen aber nicht erneut vorgeschlagen "
    "werden, wenn ihr Katalogeintrag eine Abrechnung nur einmal im Quartal oder Behandlungsfall "
    "erlaubt. Empfiehl nur dokumentierte Leistungen. Gib abschliessend die Empfehlung im "
    "vorgegebenen strukturierten Antwortschema zurueck."
)

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
            "Abrechnungsquartal: {quarter}\n\n"
            "Diktat:\n{dictation}\n\nPatientenkontext:\n{patient_context}",
        ),
    ]
)

RAG_BILLING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Du empfiehlst EBM-GOP-Abrechnungsziffern fuer eine deutsche Arztpraxis. "
            "Waehle ausschliesslich GOP-Ziffern aus den bereitgestellten Kandidaten. "
            "Denke auch an Pauschalen, wenn Patientenkontakt, Quartalsstatus und Kontext dafuer sprechen. "
            "Erfinde keine Leistungen, Diagnosen, Befunde oder GOP-Ziffern. "
            "Wenn kein Kandidat eindeutig passt, gib keine Empfehlung zurueck. "
            "Begruende jede gewaehlte Ziffer kurz.",
        ),
        (
            "human",
            "Abrechnungsquartal: {quarter}\n\n"
            "Diktat:\n{dictation}\n\nPatientenkontext:\n{patient_context}"
            "\n\nKandidaten:\n{candidates}",
        ),
    ]
)
