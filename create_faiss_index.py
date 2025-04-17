# create_faiss_index.py
import os
import time
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# --- Configurazione ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INGREDIENTS_CSV_PATH = os.path.join(DATA_DIR, "ingredients.csv")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "ingredients.index")
NAME_MAPPING_PATH = os.path.join(DATA_DIR, "ingredient_names.pkl")

# Usa lo stesso modello definito altrove
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

# --- Funzioni Ausiliarie ---


def normalize_name(name: str) -> str:
    """Normalizza il nome per il matching (minuscolo, rimuove eccesso spazi)."""
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = name.replace("  ", " ")  # Rimuove doppi spazi
    return name


def prepare_ingredient_data_enhanced(filepath: str) -> list[str]:
    """Carica e arricchisce i nomi degli ingredienti con sinonimi e varianti."""
    try:
        print(f"Caricamento nomi da: {filepath}")
        # Assumi UTF-8 o prova 'latin-1'
        df = pd.read_csv(filepath, encoding='utf-8')
        df.columns = df.columns.str.strip()
        if 'name' not in df.columns:
            raise ValueError("Colonna 'name' non trovata nel CSV.")

        # Pulisci e ottieni nomi base
        base_names = df['name'].astype(str).str.strip().str.lower()
        base_names = base_names[base_names != ''].unique().tolist()
        print(f"Trovati {len(base_names)} nomi base di ingredienti.")

        # Crea varianti linguistiche (singolare/plurale, diminutivi comuni)
        enhanced_names = []
        for name in base_names:
            enhanced_names.append(name)

            # Aggiungi singolare/plurale
            if name.endswith('e'):
                # e.g., "peperone" -> "peperoni"
                enhanced_names.append(name[:-1] + 'i')
            elif name.endswith('a'):
                # e.g., "carota" -> "carote"
                enhanced_names.append(name[:-1] + 'e')
            elif name.endswith('o'):
                # e.g., "gambero" -> "gamberi"
                enhanced_names.append(name[:-1] + 'i')

            # Aggiungi sinonimi comuni per ingredienti specifici
            common_synonyms = {
                'cuscus': ['couscous'],
                'formaggio feta': ['feta'],
                'peperone': ['peperoni', 'peperone dolce', 'peperone rosso', 'peperone giallo'],
                'gambero': ['gamberi', 'gamberetto', 'gamberetti'],
                'basilico': ['basilico fresco'],
                'coriandolo': ['coriandolo fresco', 'cilantro'],
                'melanzana': ['melanzane'],
                'olive': ['olive nere', 'olive verdi', 'olive kalamata'],
                'limone': ['lime'],
                'cipolla': ['cipolla rossa', 'cipolla bianca', 'cipolla dorata'],
                'formaggio parmigiano': ['parmigiano', 'parmigiano reggiano'],
                'pasta': ['spaghetti', 'penne', 'fusilli', 'tagliatelle', 'fettuccine'],
                'riso': ['riso bianco', 'riso integrale', 'riso arborio', 'riso carnaroli'],
                'pomodoro': ['pomodori', 'pomodorini', 'pomodoro ciliegino'],
                'funghi': ['funghi champignon', 'funghi porcini', 'champignon'],
                'zucchina': ['zucchine'],
                'mela': ['mele'],
                'pera': ['pere'],
                'arancia': ['arance'],
                'fragola': ['fragole'],
                'uva': ['uva bianca', 'uva nera', 'uva rossa'],
                'banana': ['banane'],
                'nocciola': ['nocciole'],
                'noce': ['noci'],
                'mandorla': ['mandorle'],
                'pistacchio': ['pistacchi'],
                'rosmarino': ['rosmarino fresco'],
                'timo': ['timo fresco'],
                'origano': ['origano secco', 'origano fresco'],
                'formaggio halloumi': ['halloumi']
            }

            if name in common_synonyms:
                enhanced_names.extend(common_synonyms[name])

        # Rimuovi duplicati e ordina
        unique_enhanced_names = sorted(list(set(enhanced_names)))
        print(
            f"Dataset arricchito: {len(base_names)} nomi base → {len(unique_enhanced_names)} nomi totali")

        # Salva anche la versione arricchita in un file separato per riferimento
        enhanced_mapping_path = os.path.join(
            DATA_DIR, "enhanced_ingredients.txt")
        with open(enhanced_mapping_path, 'w', encoding='utf-8') as f:
            for name in unique_enhanced_names:
                f.write(name + '\n')
        print(f"Lista arricchita salvata in: {enhanced_mapping_path}")

        return unique_enhanced_names

    except Exception as e:
        print(f"Errore durante la preparazione dei dati: {e}")
        raise


# --- Script Principale ---

if __name__ == "__main__":
    print("--- Avvio Creazione Indice FAISS Migliorato ---")

    # 1. Carica Nomi Ingredienti Arricchiti
    try:
        ingredient_names = prepare_ingredient_data_enhanced(
            INGREDIENTS_CSV_PATH)
        if not ingredient_names:
            print("Nessun nome di ingrediente trovato. Impossibile creare l'indice.")
            exit()
    except Exception as e:
        print(f"Fallimento caricamento nomi: {e}")
        exit()

    # 2. Carica Modello Embedding
    try:
        print(f"Caricamento modello SBERT: {EMBEDDING_MODEL_NAME}...")
        # Prova a usare MPS se disponibile, altrimenti CPU
        try:
            model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='mps')
            print("Modello caricato su dispositivo MPS (GPU Apple Silicon).")
        except Exception:
            print("Impossibile usare MPS, fallback su CPU.")
            model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
    except Exception as e:
        print(
            f"Errore durante il caricamento del modello SentenceTransformer: {e}")
        exit()

    # 3. Calcola Embeddings
    try:
        print(f"Calcolo embeddings per {len(ingredient_names)} nomi...")
        start_time = time.time()
        embeddings = model.encode(
            ingredient_names,
            convert_to_numpy=True,
            show_progress_bar=True,
            # IMPORTANTE: Normalizza per usare IndexFlatIP (Cosine Similarity)
            normalize_embeddings=True
        )
        # Assicurati che siano float32 (preferito da FAISS)
        embeddings = embeddings.astype('float32')
        end_time = time.time()
        print(f"Embeddings calcolati in {end_time - start_time:.2f} secondi.")
        print(f"Shape matrice embeddings: {embeddings.shape}")
    except Exception as e:
        print(f"Errore durante il calcolo degli embeddings: {e}")
        exit()

    # 4. Crea e Popola Indice FAISS
    try:
        dimension = embeddings.shape[1]  # Dimensione degli embedding
        print(
            f"Creazione indice FAISS (IndexFlatIP) con dimensione {dimension}...")
        # IndexFlatIP calcola il prodotto scalare. Poiché abbiamo normalizzato
        # gli embedding (norma L2 = 1), il prodotto scalare è equivalente
        # alla similarità cosenus. Valori più alti indicano maggiore similarità.
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        print(f"Indice creato e popolato con {index.ntotal} vettori.")
    except Exception as e:
        print(f"Errore durante la creazione dell'indice FAISS: {e}")
        exit()

    # 5. Salva Indice e Mapping Nomi
    try:
        print(f"Salvataggio indice FAISS in: {FAISS_INDEX_PATH}")
        faiss.write_index(index, FAISS_INDEX_PATH)

        print(f"Salvataggio mapping nomi in: {NAME_MAPPING_PATH}")
        with open(NAME_MAPPING_PATH, 'wb') as f:
            # Salva la lista di nomi nello stesso ordine
            pickle.dump(ingredient_names, f)

        print("--- Indice FAISS e Mapping Nomi Creati con Successo! ---")

    except Exception as e:
        print(f"Errore durante il salvataggio dei file: {e}")
        exit()

    # 6. Verifica Rapidamente la Qualità dell'Indice
    try:
        print("\n--- Test Rapido dell'Indice Creato ---")
        # Test alcuni ingredienti problematici
        test_ingredients = [
            "gamberi", "peperoni", "couscous", "feta", "olive nere",
            "basilico", "carote", "halloumi", "coriandolo"
        ]

        for test_ing in test_ingredients:
            test_emb = model.encode(
                [test_ing],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype('float32')

            D, I = index.search(test_emb, 3)  # Ottieni i top 3 match

            print(f"\nTest ingrediente: '{test_ing}'")
            for i in range(3):
                match_idx = I[0][i]
                match_score = D[0][i]
                match_name = ingredient_names[match_idx]
                print(
                    f"  Match #{i+1}: '{match_name}' (Score: {match_score:.4f})")

    except Exception as e:
        print(f"Errore durante il test dell'indice: {e}")
        # Non terminiamo il programma qui in quanto il test è opzionale

    print("\n--- Processo Completato ---")
