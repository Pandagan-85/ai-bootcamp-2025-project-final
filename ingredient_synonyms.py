# ingredient_synonyms.py

"""
Dizionario centralizzato di sinonimi e mappature per ingredienti.
Utilizzato sia per la creazione dell'indice FAISS che per il verificatore.
Contiene:
- SYNONYMS_FOR_INDEX: Usato per arricchire l'indice FAISS
- FALLBACK_MAPPING: Usato dal verificatore come strategia di fallback
- INVARIABLE_WORDS: Parole che non cambiano al singolare/plurale
- ALWAYS_PLURAL: Parole che sono sempre al plurale
"""

# Parole invariabili (non cambiano al plurale)
INVARIABLE_WORDS = {
    'latte', 'pepe', 'formaggio', 'sale', 'olio', 'aceto', 'caffè', 'tè',
    'miele', 'marmellata', 'pane', 'cous cous', 'curry', 'senape', 'maionese',
    'ketchup', 'wasabi', 'miglio', 'quinoa', 'orzo', 'burro', 'farro', 'yogurt',
    'riso', 'grano', 'semola', 'amido', 'zucchero', 'farina', 'mais', 'lievito',
    'brodo', 'concentrato di pomodoro', 'passata di pomodoro', 'pelati', 'acqua', 'lattuga'
}

# Parole che sono già al plurale
ALWAYS_PLURAL = {
    'olive', 'alici', 'acciughe', 'anacardi', 'arachidi', 'capperi', 'rognoni',
    'marsala', 'funghi', 'spinaci', 'asparagi', 'lenticchie', 'fagioli', 'ceci',
    'piselli', 'pinoli', 'pistacchi', 'mandorle', 'noci', 'nocciole', 'datteri',
    'fichi', 'prugne', 'uvetta', 'albicocche', 'molluschi', 'frutti di mare',
    'gamberetti', 'cozze', 'vongole', 'calamari', 'seppie', 'broccoli'
}

# Formato {nome_normalizzato: [lista_varianti]}
# Chiave: nome CSV norm. Valore: varianti norm.
# Usato da create_faiss_index.py per arricchire l'indice
SYNONYMS_FOR_INDEX = {
    'couscous': ['cuscus'],
    'linguine': ['pasta linguine'],
    'peperoni': ['peperone', 'peperone dolce', 'peperone rosso', 'peperone giallo', 'peperoni rossi', 'peperoni gialli'],
    'gamberi': ['gambero', 'gamberetto', 'gamberetti'],
    'basilico': ['basilico fresco', 'foglie di basilico fresco'],
    'menta': ['menta fresca', 'foglie di menta'],
    'coriandolo': ['coriandolo fresco', 'cilantro'],
    'spezie miste': ['cumino', 'coriandolo', 'paprika'],
    'melanzane': ['melanzana'],
    'mandorle': ['mandorle a scaglie', 'mandorle a lamelle', 'mandorle tritate', 'mandorle a fette'],
    'pancetta': ['pancetta a cubetti'],
    'pecorino': ['pecorino grattugiato', 'pecorino romano grattugiato'],
    'cetriolo': ['cetrioli', 'cetriolo a cubetti', 'cetrioli a cubetti'],

    # Chiave: nome CSV norm. Valore: varianti norm.
    'limone': ['limoni', 'scorza di limone', 'scorza di limone grattugiata', 'lievi scorze di limone grattugiate', 'lime'],
    'vino bianco': ['vino bianco secco'],
    'cipolla': ['cipolla rossa', 'cipolla bianca', 'cipolla dorata'],
    'parmigiano reggiano': ['parmigiano', 'formaggio parmigiano', 'parmigiano grattugiato'],
    'spaghetti': ['pasta spaghetti', 'pasta'],
    'penne': ['pasta penne', 'pasta'],
    'fusilli': ['pasta fusilli', 'pasta'],
    'tagliatelle': ['pasta tagliatelle'],
    'riso basmati': ['riso', 'riso bianco'],
    'riso jasmine': ['riso', 'riso bianco'],
    'riso venere': ['riso', 'riso bianco'],
    'riso arborio': ['riso', 'riso bianco'],
    'pomodoro': ['pomodori', 'pomodoro fresco', 'pomodori freschi', 'pomodoro a cubetti'],
    'funghi': ['funghi champignon', 'funghi porcini', 'champignon'],
    'mirtilli': ['mirtilli rossi'],
    'uvetta': ['uvetta sultanina', 'uvetta comune'],
    'zucchine': ['zucchina'],
    'mela': ['mele'],
    'pera': ['pere'],
    'arancia': ['arance'],
    'fragole': ['fragola'],
    'uva': ['uva bianca', 'uva nera', 'uva rossa'],  # Forme corrette
    'banana': ['banane'],
    'rosmarino': ['rosmarino fresco'],
    'timo': ['timo fresco'],
    'origano': ['origano secco', 'origano fresco'],

    'polpo': ['polipo'],
    'rucola': ['rughetta', 'rucola fresca'],
    'feta': ['feta a cubetti', 'formaggio feta'],
    'zafferano': ['zafferano in polvere'],
    'prezzemolo': ['prezzemolo fresco', 'prezzemolo in polvere', 'prezzemolo secco'],
    'lenticchie': ['lenticchie rosse', 'lenticchie verdi'],
    'pasta di lenticchie': ['pasta di lenticchie rosse', 'pasta di lenticchie verdi'],
    'curcuma': ['curmuca fresca']
}

# Formato {variante_normalizzata: nome_originale_csv}
# Chiave: variante norm. Valore: nome esatto CSV
# Usato da verifier_agent.py come fallback (quando non trova nulla nell'indice)
FALLBACK_MAPPING = {
    "pomodori": "Pomodoro",  # Nome esatto nel CSV
    "pomodoro fresco": "Pomodoro",
    "pomodori freschi": "Pomodoro",
    "pomodoro a cubetti": "Pomodoro",
    "pomodori a cubetti": "Pomodoro",
    "peperoni": "Peperoni",
    "peperone": "Peperoni",
    "peperone rosso": "Peperoni",
    "pecorino grattugiato": "Pecorino",
    "pecorino romano grattugiato": "Pecorino",
    "pancetta a cubetti": "Pancetta",
    "cetrioli": "Cetriolo",
    "cetriolo a cubetti": "Cetriolo",
    "cetrioli a cubetti": "Cetriolo",
    "feta a cubetti": "Feta",
    "mandorle a scaglie": "Mandorle",
    "mandorle a lamelle": "Mandorle",
    "mandorle a fette": "Mandorle",
    "mandorle tritate": "Mandorle",
    "tagliolini": "Tagliatelle",
    "tagliolini freschi": "Tagliatelle",
    "rucola fresca": "Rucola",
    "rughetta": "Rucola",
    "cipolla rossa": "Cipolla",
    "cipolla bianca": "Cipolla",
    "cipolla dorata": "Cipolla",

    "zucchina": "Zucchine",
    "zucchini": "Zucchine",
    "pepe": "Pepe nero",
    "curcuma": "Curcuma",
    "coriandolo": "Coriandolo",
    "coriandolo fresco": "Coriandolo",
    "origano fresco": "Origano",
    "origano secco": "Origano",
    "mirtilli rossi": "Mirtilli",
    "prezzemolo fresco": "Prezzemolo",
    "spaghetti": "Spaghetti",
    "zenzero fresco": "Zenzero",
    "zenzero fresco grattugiato": "Zenzero",
    "limoni": "Limone",  # Chiave: variante norm. Valore: nome esatto CSV
    "scorza di limone": "Limone",  # Mappiamo anche la scorza al limone generico
    "scorza di limone grattugiata": "Limone",
    "lievi scorze di limone grattugiate": "Limone",
    "lime": "Limone",
    "uvetta sultanina": "Uvetta",
    "vino bianco secco": "Vino bianco"
}

# Funzione helper per generare automaticamente FALLBACK_MAPPING da SYNONYMS_FOR_INDEX
# da testare appena funziona veramente tutto


def generate_fallback_mapping(synonyms_dict, csv_names_mapping):
    """
    Genera automaticamente il dizionario di fallback partendo dal dizionario di sinonimi.

    Args:
        synonyms_dict: Dizionario nel formato {nome_normalizzato: [lista_varianti]}
        csv_names_mapping: Dizionario nel formato {nome_normalizzato: nome_originale_nel_csv}

    Returns:
        Dizionario nel formato {variante_normalizzata: nome_originale_nel_csv}
    """
    fallback = {}
    for main_name, variants in synonyms_dict.items():
        original_csv_name = csv_names_mapping.get(main_name)
        if not original_csv_name:
            continue

        # Aggiungi mapping per le varianti
        for variant in variants:
            fallback[variant] = original_csv_name

    return fallback

# Funzione per la normalizzazione (duplicata da utils.py per convenienza)


def normalize_for_synonyms(name):
    """Versione semplificata di normalize_name per uso nei sinonimi."""
    return name.lower().strip()
