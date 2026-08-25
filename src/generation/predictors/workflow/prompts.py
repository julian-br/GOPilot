from langchain_core.prompts import ChatPromptTemplate

UNDERSTANDING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extrahiere alle ausdruecklich erbrachten aerztlichen Leistungen und Kontaktformen. "
            "Jedes Objekt enthaelt genau eine Leistung und benennt ausschliesslich deren Art. "
            "Erlaeutere oder fasse nicht zusammen, was konkret gesagt, besprochen, berichtet, "
            "gemessen, festgestellt oder empfohlen wurde. Nenne keine Diagnosen, Werte, "
            "Ergebnisse oder sonstigen Fallinhalte. Fasse Duplikate und Teilschritte derselben "
            "Leistung zusammen und erfinde nichts.",
        ),
        ("human", "Diktat:\n{dictation}"),
    ]
)

SERVICE_SELECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Pruefe die Suchergebnisse fuer eine Leistung streng gegen das Diktat. Waehle "
            "einen Code nur, wenn die beschriebene Leistung tatsaechlich erbracht wurde und die "
            "im Katalogtext erkennbaren Voraussetzungen passen. Diagnosen, Messwerte, "
            "thematische Aehnlichkeit oder der Retrieval-Rang allein genuegen nicht. Bewerte "
            "den gesamten Katalogtext, alle Anmerkungen und Abrechnungsregeln. Eine allgemeine "
            "Teilleistung genuegt nicht, wenn eine spezifische Indikation, Zielgruppe oder "
            "GOP-Verknuepfung fehlt. Jede spezifische Voraussetzung braucht einen positiven "
            "Beleg; fehlende Angaben duerfen nicht angenommen werden. Durch Verweise definierte "
            "Begriffe duerfen nicht nach ihrer allgemeinen medizinischen Bedeutung umgedeutet "
            "werden. Ein erwaehnter oder beurteilter Befund belegt nicht die Durchfuehrung der "
            "zugrunde liegenden Untersuchung. Begruendungen duerfen nur ausdruecklich "
            "dokumentierte Fakten enthalten. Diagnosen, Dauer, Qualifikationen und andere "
            "Voraussetzungen duerfen nicht abgeleitet werden. Nutze Fallkontext und Historie "
            "nur fuer Voraussetzungen und Ausschluesse, nicht als Beleg fuer eine aktuell "
            "erbrachte Leistung. Gib nur plausible Codes aus den Suchergebnissen zurueck; es "
            "duerfen auch alle verworfen werden.",
        ),
        (
            "human",
            "Fallkontext und Historie:\n{case_context}\n\n"
            "Aktuelles Diktat:\n{dictation}\n\n"
            "Leistung:\n{service}\n\n"
            "Suchergebnisse:\n{candidates}",
        ),
    ]
)

FLAT_SELECTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Pruefe die Suchergebnisse als moegliche Pauschalen oder kontextabhaengige "
            "Zuschlaege streng gegen Diktat, Fallkontext und Historie. Waehle nur Codes, deren "
            "im Katalogtext erkennbaren Voraussetzungen positiv dokumentiert sind. Fehlende "
            "Angaben duerfen nicht angenommen werden. Strukturierte Diagnosen und Kontakte "
            "sind dokumentierte Fakten. Einen Ausschluss wendest du nur an, wenn der Fall einen "
            "positiven Hinweis auf den Ausschlusstatbestand enthaelt; seine Nichtanwendbarkeit "
            "muss nicht ausdruecklich dokumentiert sein. Beachte GOP-Verknuepfungen, "
            "Anmerkungen und Abrechnungsregeln. Ein Behandlungsfall ist niemals mit einem "
            "einzelnen Kontakt gleichzusetzen. Pruefe bereits gebuchte GOPs im jeweiligen "
            "Bezugsraum. Bewerte voneinander abhaengige Kandidaten gemeinsam. Waehle keine "
            "Einzelleistungen, auch wenn diese im Fall erbracht wurden. Die Reihenfolge der "
            "Kandidaten drueckt keine Relevanz aus. Diagnosen oder thematische Aehnlichkeit "
            "allein genuegen nicht. Begruendungen duerfen nur dokumentierte Fakten enthalten. "
            "Gib nur positiv belegte Empfehlungen aus; es duerfen auch alle Kandidaten "
            "verworfen werden. Begruende jede Empfehlung kurz.",
        ),
        (
            "human",
            "Fallkontext und Historie:\n{case_context}\n\n"
            "Begriffsdefinitionen:\n{definitions}\n\n"
            "Aktuelles Diktat:\n{dictation}\n\n"
            "Kandidaten:\n{candidates}",
        ),
    ]
)
