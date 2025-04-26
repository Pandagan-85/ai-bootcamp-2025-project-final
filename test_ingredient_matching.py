"""
Script per testare il matching di ingredienti con il sistema FAISS.

Questo script permette di:
1. Testare il matching di ingredienti specifici
2. Verificare come verranno interpretati gli ingredienti generati dall'LLM
3. Visualizzare i dettagli nutrizionali e dietetici degli ingredienti matchati
"""

from utils import find_best_match_faiss, normalize_name
from loaders import load_ingredient_database_with_mappings
from ingredient_synonyms import FALLBACK_MAPPING
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Percorsi dei file (adatta in base alla tua struttura)
INGREDIENTS_FILE = "data/ingredients.csv"
FAISS_INDEX_FILE = "data/ingredients.index"
INDEX_TO_NAME_FILE = "data/ingredient_names.pkl"

# Carica database ingredienti e mappature
ingredient_data, normalized_to_original, original_to_normalized = load_ingredient_database_with_mappings(
    INGREDIENTS_FILE)

# Funzione per caricare l'indice FAISS e il modello


def load_faiss_resources(index_path, mapping_path):
    """
    Carica l'indice FAISS, il mapping dei nomi e il modello di embedding.

    Args:
        index_path: Percorso al file dell'indice FAISS
        mapping_path: Percorso al file del mapping dei nomi

    Returns:
        Tupla (faiss_index, index_to_name_mapping, embedding_model)
    """
    print(f"Caricamento indice FAISS da {index_path}...")
    faiss_index = faiss.read_index(index_path)

    print(f"Caricamento mapping nomi da {mapping_path}...")
    with open(mapping_path, 'rb') as f:
        index_to_name_mapping = pickle.load(f)

    print(f"Caricamento modello SentenceTransformer...")
    embedding_model = SentenceTransformer(
        'paraphrase-multilingual-mpnet-base-v2')

    return faiss_index, index_to_name_mapping, embedding_model


# Carica indice FAISS e modello
faiss_index, index_to_name_mapping, embedding_model = load_faiss_resources(
    FAISS_INDEX_FILE, INDEX_TO_NAME_FILE)


def test_ingredient_matching(ingredient_name, threshold=0.60):
    """
    Testa il matching di un singolo ingrediente e mostra i risultati.

    Args:
        ingredient_name: Nome dell'ingrediente da testare
        threshold: Soglia di similarità (default: 0.60)
    """
    print(f"\n--- Matching test per '{ingredient_name}' ---")

    # Normalizza il nome per debug
    normalized_name = normalize_name(ingredient_name)
    print(f"Nome normalizzato: '{normalized_name}'")

    # Cerca il miglior match
    match_result = find_best_match_faiss(
        llm_name=ingredient_name,
        faiss_index=faiss_index,
        index_to_name_mapping=index_to_name_mapping,
        model=embedding_model,
        normalize_func=normalize_name,
        threshold=threshold
    )

    if match_result:
        matched_db_name, match_score = match_result
        print(f"Miglior match: '{matched_db_name}' (score: {match_score:.4f})")

        # Verifica se il match è presente nel database
        if matched_db_name in ingredient_data:
            print(f"✅ Presente nel database")
            info = ingredient_data[matched_db_name]
            print(f"  CHO per 100g: {info.cho_per_100g}")
            print(
                f"  Vegan: {info.is_vegan}, Vegetarian: {info.is_vegetarian}")
            print(
                f"  Gluten-Free: {info.is_gluten_free}, Lactose-Free: {info.is_lactose_free}")
        else:
            print(f"❌ NON presente nel database")

            # Verifica normalizzazione e mapping
            if normalized_name in normalized_to_original:
                original_name = normalized_to_original[normalized_name]
                print(f"  Tramite normalizzazione → '{original_name}'")
                if original_name in ingredient_data:
                    print(f"  ✅ Trovato tramite normalizzazione")

            # Verifica fallback mapping
            if normalized_name in FALLBACK_MAPPING:
                fallback_name = FALLBACK_MAPPING[normalized_name]
                print(f"  Presente nel fallback mapping → '{fallback_name}'")
                if fallback_name in ingredient_data:
                    print(f"  ✅ Trovato tramite fallback mapping")
    else:
        print("❌ Nessun match trovato con threshold corrente")

        # Verifica fallback mapping per ingredienti senza match
        if normalized_name in FALLBACK_MAPPING:
            fallback_name = FALLBACK_MAPPING[normalized_name]
            print(f"  Presente nel fallback mapping → '{fallback_name}'")
            if fallback_name in ingredient_data:
                print(f"  ✅ Trovato tramite fallback mapping")

        # Suggerimento per test con threshold più bassa
        print("  Prova a ridurre la threshold per vedere match potenziali")


def test_multiple_ingredients(ingredients_list):
    """
    Testa una lista di ingredienti e mostra statistiche riassuntive.

    Args:
        ingredients_list: Lista di nomi di ingredienti da testare
    """
    print(f"\n=== Testando {len(ingredients_list)} ingredienti ===")

    matched_count = 0
    for i, ing in enumerate(ingredients_list, 1):
        print(f"\n[{i}/{len(ingredients_list)}]", end="")
        test_ingredient_matching(ing)

        # Conta quanti match abbiamo ottenuto
        match_result = find_best_match_faiss(
            llm_name=ing,
            faiss_index=faiss_index,
            index_to_name_mapping=index_to_name_mapping,
            model=embedding_model,
            normalize_func=normalize_name,
            threshold=0.60
        )
        if match_result:
            matched_db_name, _ = match_result
            if matched_db_name in ingredient_data:
                matched_count += 1

    # Statistiche finali
    print(f"\n=== Risultati ===")
    print(f"Ingredienti testati: {len(ingredients_list)}")
    print(
        f"Match trovati: {matched_count} ({matched_count/len(ingredients_list)*100:.1f}%)")
    print(f"Match mancanti: {len(ingredients_list) - matched_count}")


# Test con alcuni ingredienti predefiniti
ingredienti_test = [
    "Farina",
    "Farina integrale",
    "Patate",
    "Patate dolci",
    "Zucchine",
    "Pomodori ciliegini",
    "Olio di oliva",
    "pane grattugiato",
    "Ceci in scatola (sciacquati e sgocciolati)"
]

# Esegui i test
print("\n\n======== BATCH TEST =========")
test_multiple_ingredients(ingredienti_test)

# Modalità interattiva
print("\n\n======== MODALITÀ INTERATTIVA =========")
print("Inserisci un ingrediente alla volta per testare il matching")
print("(inserisci 'q' per uscire, 'batch' per testare una lista di ingredienti)")

while True:
    test_input = input(
        "\nInserisci un ingrediente da testare (o 'q' per uscire): ")
    if test_input.lower() == 'q':
        break
    elif test_input.lower() == 'batch':
        # Permette di testare una lista di ingredienti
        print("Inserisci gli ingredienti uno per riga. Linea vuota per terminare:")
        batch_ingredients = []
        while True:
            line = input()
            if not line.strip():
                break
            batch_ingredients.append(line.strip())

        if batch_ingredients:
            test_multiple_ingredients(batch_ingredients)
    else:
        test_ingredient_matching(test_input)
