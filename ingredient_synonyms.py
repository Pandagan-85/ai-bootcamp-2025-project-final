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
    'peperoni': ['peperone', 'peperone dolce', 'peperone rosso', 'peperone giallo', 'peperoni rossi', 'peperoni gialli'],
    'ceci cotti': ['ceci in scatola', 'ceci in lattina', 'ceci secchi', 'ceci lessati', 'ceci precotti', 'ceci in scatola (sciacquati e sgocciolati)', 'ceci sciacquati', 'ceci scolati'],
    'gamberi': ['gambero', 'gamberetto', 'gamberetti', 'gamberetti sgusciati'],
    'basilico': ['basilico fresco', 'foglie di basilico fresco'],
    'menta': ['menta fresca', 'foglie di menta'],
    'coriandolo': ['coriandolo fresco', 'cilantro'],
    'cumino': ['cumino in polvere', 'cumino macinato', 'semi di cumino', 'polvere di cumino'],
    'spezie miste': ['mix di spezie', 'spezie assortite', 'spezie miste (paprika, cumino, pepe nero)',
                     'mix di erbe aromatiche', 'erbe aromatiche miste', 'spezie per curry'],
    'paprika': ['paprica', 'peperoncino in polvere dolce'],
    'melanzane': ['melanzana'],
    'mandorle': ['mandorle a scaglie', 'mandorla', 'mandorla a scaglie', 'mandorle a lamelle', 'mandorle tritate', 'mandorle a fette', 'mandorle a fettine', 'mandorle tostate', 'mandorle pelate', 'granella di mandorle', 'mandorle affettate'],
    'pancetta': ['pancetta a cubetti'],
    'pecorino': ['pecorino grattugiato', 'pecorino romano grattugiato'],
    'cetriolo': ['cetrioli', 'cetriolo a cubetti', 'cetrioli a cubetti'],
    'zenzero': ['zenzero fresco', 'zenzero fresco grattugiato'],
    # Chiave: nome CSV norm. Valore: varianti norm.
    'limone': ['limoni', 'scorza di limone', 'scorza di limone grattugiata', 'lievi scorze di limone grattugiate', 'lime'],
    'cipolla': ['cipolla rossa', 'cipolla bianca', 'cipolla dorata'],
    'parmigiano reggiano': ['parmigiano', 'formaggio parmigiano', 'parmigiano grattugiato'],
    'linguine': ['pasta linguine', 'pasta (linguine)', 'linguina'],
    'spaghetti': ['pasta spaghetti', 'pasta', 'pasta (spaghetti)'],
    'penne': ['pasta penne', 'pasta', 'pasta (penne)'],
    'fusilli': ['pasta fusilli', 'pasta', 'pasta (fusilli)'],
    'tagliatelle': ['pasta tagliatelle', 'pasta (tagliatelle)'],
    'riso bianco': ['riso'],
    'riso basmati': ['riso'],
    'riso jasmine': ['riso'],
    'riso venere': ['riso'],
    'riso arborio': ['riso'],
    'pomodoro': ['pomodori', 'pomodoro fresco', 'pomodori freschi', 'pomodoro a cubetti'],
    'funghi': ['funghi champignon', 'funghi porcini', 'champignon'],
    'mirtilli': ['mirtilli rossi'],
    'uvetta': ['uvetta sultanina', 'uvetta comune'],
    'zucchine': ['zucchina'],
    'mela': ['mele'],
    'pera': ['pere'],
    'aglio': ['aglio in polvere'],
    'pepe nero': ['pepe nero macinato', 'pepe', 'pepe nero macinato', 'pepe nero in grani', 'sale e pepe'],
    'arancia': ['arance'],
    'fragole': ['fragola'],
    'uva': ['uva bianca', 'uva nera', 'uva rossa'],  # Forme corrette
    'banana': ['banane'],
    'rosmarino': ['rosmarino fresco'],
    'timo': ['timo fresco', 'timo secco', 'timo essiccato'],
    'origano': ['origano secco', 'origano fresco'],
    'pesche': ['pesca', 'pesca fresca'],
    'polpo': ['polipo'],
    'rucola': ['rughetta', 'rucola fresca'],
    'feta': ['feta a cubetti', 'formaggio feta', 'feta sbriciolata'],
    'zafferano': ['zafferano in polvere'],
    'prezzemolo': ['prezzemolo fresco', 'prezzemolo in polvere', 'prezzemolo secco', 'prezzemolo tritato'],
    'lenticchie': ['lenticchie rosse', 'lenticchie verdi'],
    'pasta di lenticchie': ['pasta di lenticchie rosse', 'pasta di lenticchie verdi'],
    'curcuma': ['curcuma in polvere', 'polvere di curcuma', 'curcuma fresca', 'curcuma macinata'],
    'finocchio': ['finocchi', 'finocchio fresco', 'bulbo di finocchio'],
    'vino bianco': ['vino bianco secco', 'vino bianco dolce', 'vino bianco da cucina'],
    'vino rosso': ['vino rosso secco', 'vino rosso corposo', 'vino rosso da cucina'],
    'sale': ['sale fino', 'sale grosso', 'sale marino', 'sale himalayano', 'sale e pepe'],
    'uova': ['uovo', 'uovo sbattuto', 'uovo intero', 'albume', 'tuorlo', 'uova intere'],

}

# Formato {variante_normalizzata: nome_originale_csv}
# Chiave: variante norm. Valore: nome esatto CSV
# Usato da verifier_agent.py come fallback (quando non trova nulla nell'indice)
FALLBACK_MAPPING = {
    "pomodori": "Pomodoro",  # Nome esatto nel CSV
    "pomodoro fresco": "Pomodoro",
    "pane grattugiato": "Pangrattato",
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
    "finocchi": "Finocchio",
    "finocchio": "Finocchio",
    "feta a cubetti": "Feta",
    "feta sbriciolata": "Feta",
    "mandorle affettate": "Mandorle",
    "mandorle a scaglie": "Mandorle",
    "mandorla a scaglie": "Mandorle",
    "mandorle a lamelle": "Mandorle",
    "mandorle a fette": "Mandorle",
    "mandorle a fettine": "Mandorle",
    "mandorle tritate": "Mandorle",
    "mandorle tostate": "Mandorle",
    "mandorle pelate": "Mandorle",
    "granella di mandorle": "Mandorle",
    "menta fresca": "Menta",
    "basilico": "Basilico",
    "basilico fresco": "Basilico",
    "foglie di basilico fresco": "Basilico",
    "tagliolini": "Tagliatelle",
    "tagliolini freschi": "Tagliatelle",
    "aglio in polvere": "Aglio",
    "aglio": "Aglio",
    "rucola fresca": "Rucola",
    "rughetta": "Rucola",
    "cipolla rossa": "Cipolla",
    "cipolla bianca": "Cipolla",
    "cipolla dorata": "Cipolla",

    "zucchina": "Zucchine",
    "zucchini": "Zucchine",
    "pepe": "Pepe nero",
    "pepe nero macinato": "Pepe nero",
    "curcuma": "Curcuma",
    "curcuma in polvere": "Curcuma",
    "polvere di curcuma": "Curcuma",
    "curcuma macinata": "Curcuma",
    "curcuma fresca": "Curcuma",
    "coriandolo": "Coriandolo",
    "coriandolo fresco": "Coriandolo",
    "origano fresco": "Origano",
    "origano secco": "Origano",
    "mirtilli rossi": "Mirtilli",
    "prezzemolo fresco": "Prezzemolo",
    "prezzemolo in polvere": "Prezzemolo",
    "prezzemolo tritato": "Prezzemolo",
    "spaghetti": "Spaghetti",
    "pasta (linguine)": "Linguine",
    "pasta linguine": "Linguine",
    "linguine": "Linguine",
    "linguina": "Linguine",
    "pasta (spaghetti)": "Spaghetti",
    "pasta spaghetti": "Spaghetti",
    "pasta (penne)": "Penne",
    "pasta penne": "Penne",
    "pasta (fusilli)": "Fusilli",
    "pasta fusilli": "Fusilli",
    "pasta (tagliatelle)": "Tagliatelle",
    "pasta tagliatelle": "Tagliatelle",
    "zenzero fresco": "Zenzero",
    "zenzero fresco grattugiato": "Zenzero",
    "timo fresco": "Timo",
    "timo secco": "Timo",
    "timo essiccato": "Timo",
    "limoni": "Limone",  # Chiave: variante norm. Valore: nome esatto CSV
    "scorza di limone": "Limone",  # Mappiamo anche la scorza al limone generico
    "scorza di limone grattugiata": "Limone",
    "lievi scorze di limone grattugiate": "Limone",
    "lime": "Limone",
    "uvetta sultanina": "Uvetta",
    "gamberetti": "Gamberi",
    "gamberetti sgusciati": "Gamberi",
    "rosmarino fresco": "Rosmarino",
    "zafferano in polvere": "Zafferano",
    "sale e pepe": "Sale",  # Mappare a sale per semplicità
    "sale fino": "Sale",
    "sale grosso": "Sale",
    "sale marino": "Sale",
    "sale himalayano": "Sale",
    "uovo": "Uova",
    "uova": "Uova",
    "uovo sbattuto": "Uova",
    "uovo intero": "Uova",
    "uova intere": "Uova",
    "vino bianco": "Vino bianco",
    "vino bianco secco": "Vino bianco",
    "vino bianco dolce": "Vino bianco",
    "vino bianco da cucina": "Vino bianco",
    "vino rosso": "Vino rosso",
    "vino rosso secco": "Vino rosso",
    "vino rosso corposo": "Vino rosso",
    "vino rosso da cucina": "Vino rosso",
    "cumino in polvere": "Cumino",
    "cumino macinato": "Cumino",
    "semi di cumino": "Cumino",
    "polvere di cumino": "Cumino",
    "ceci": "Ceci cotti",
    "ceci in scatola": "Ceci cotti",
    "ceci in lattina": "Ceci cotti",
    "ceci lessati": "Ceci cotti",
    "ceci precotti": "Ceci cotti",
    "ceci in scatola (sciacquati e sgocciolati)": "Ceci cotti",
    "ceci in scatola (sciacquati e scolati)": "Ceci cotti",
    "ceci sciacquati": "Ceci cotti",
    "ceci scolati": "Ceci cotti",
    "hummus di ceci": "Hummus",
    "humus di ceci": "Hummus",
    "humus": "Hummus"
}

INCOMPATIBLE_MATCHES = [
    # Formati come tuple: ("ingrediente1", "ingrediente2")
    # dove l'ingrediente1 non dovrebbe mai essere mappato a ingrediente2
    ("mandorle", "pancetta"),
    ("pancetta", "mandorle"),
    ("curcuma", "prezzemolo"),
    ("prezzemolo", "curcuma"),
    ("mandorle", "peperoni"),
    ("peperoni", "mandorle"),
    ("pepe nero", "peperone"),
    ("peperone", "pepe nero"),
    ("pepe", "peperone"),
    ("peperone", "pepe"),
    # Aggiungi altre coppie incompatibili qui
]

# Funzione per verificare se una corrispondenza è incompatibile


def is_incompatible_match(ingredient1, ingredient2):
    """
    Verifica se la corrispondenza tra due ingredienti è incompatibile.

    Args:
        ingredient1: Nome normalizzato del primo ingrediente
        ingredient2: Nome normalizzato del secondo ingrediente

    Returns:
        True se la corrispondenza è incompatibile, False altrimenti
    """
    # Normalizza entrambi gli ingredienti
    ing1 = normalize_for_synonyms(ingredient1)
    ing2 = normalize_for_synonyms(ingredient2)

    # Controlla se c'è una incompatibilità diretta
    for incomp1, incomp2 in INCOMPATIBLE_MATCHES:
        if (ing1.startswith(incomp1) and ing2.startswith(incomp2)) or \
           (ing1.startswith(incomp2) and ing2.startswith(incomp1)):
            return True

    return False

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
