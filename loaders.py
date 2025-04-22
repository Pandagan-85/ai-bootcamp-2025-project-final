# loaders.py
import time
import os
from typing import List, Dict, Any, Callable
import pandas as pd
import numpy as np
# SentenceTransformer NON è più necessario qui per caricare i dati base
from utils import normalize_name  # Assumi sia ancora in utils
from model_schema import IngredientInfo


def load_ingredient_database_with_mappings(filepath: str):
    """
    Carica il database degli ingredienti e crea mappature normalizzate.

    Args:
        filepath: Percorso al file CSV degli ingredienti

    Returns:
        Tuple con:
        - ingredient_data: Dizionario originale {nome_originale: IngredientInfo}
        - normalized_to_original: Dizionario {nome_normalizzato: nome_originale}
        - original_to_normalized: Dizionario {nome_originale: nome_normalizzato}
    """
    # Carica il database degli ingredienti usando la funzione esistente
    ingredient_data = load_basic_ingredient_info(filepath)

    # Debug
    print(
        f"Caricati {len(ingredient_data) if ingredient_data else 0} ingredienti dal database CSV")
    if ingredient_data:
        sample_keys = list(ingredient_data.keys())[:5]
        print(f"Esempio primi 5 ingredienti: {sample_keys}")

    if not ingredient_data:
        print("ERRORE: Nessun dato di ingredienti caricato dal CSV!")
        return {}, {}, {}

    # Crea mappature normalizzate
    normalized_to_original = {}
    original_to_normalized = {}

    # DEBUG: Stampa alcune chiavi del dizionario per verificare
    print(f"DEBUG: Creazione mappature per {len(ingredient_data)} ingredienti")
    debug_count = 0

    for original_name, info in ingredient_data.items():
        # Assicurati che original_name sia una stringa valida
        if not isinstance(original_name, str):
            print(
                f"ATTENZIONE: Chiave non stringa nel dizionario: {type(original_name)}")
            continue

        normalized_name = normalize_name(original_name)

        # DEBUG per verificare la normalizzazione
        if debug_count < 5:
            print(f"DEBUG: Mappatura '{original_name}' -> '{normalized_name}'")
            debug_count += 1

        # Gestione di potenziali collisioni (due nomi diversi che normalizzano allo stesso valore)
        if normalized_name in normalized_to_original:
            existing_original = normalized_to_original[normalized_name]
            print(f"ATTENZIONE (loaders): Collisione nella normalizzazione! "
                  f"'{original_name}' e '{existing_original}' normalizzano a '{normalized_name}'")
            # Mantieni comunque la mappatura (la prima vince in caso di collisione)
        else:
            normalized_to_original[normalized_name] = original_name

        # La mappatura inversa è sempre 1:1
        original_to_normalized[original_name] = normalized_name

    print(
        f"Mappature di normalizzazione create: {len(normalized_to_original)} uniche, {len(original_to_normalized)} originali")

    # DEBUG: Verifica alcune mappature create
    if len(normalized_to_original) > 0:
        sample_keys = list(normalized_to_original.keys())[:3]
        print(f"DEBUG: Esempio mappature normalizzate:")
        for key in sample_keys:
            print(f"  '{key}' -> '{normalized_to_original[key]}'")
    else:
        # Aggiunto questo DEBUG
        print("DEBUG: normalized_to_original è vuoto!")

    if len(original_to_normalized) > 0:
        sample_keys_orig = list(original_to_normalized.keys())[:3]
        print(f"DEBUG: Esempio mappature originali:")
        for key in sample_keys_orig:
            print(f"  '{key}' -> '{original_to_normalized[key]}'")
    else:
        # Aggiunto questo DEBUG
        print("DEBUG: original_to_normalized è vuoto!")

    return ingredient_data, normalized_to_original, original_to_normalized


def load_basic_ingredient_info(filepath: str) -> Dict[str, IngredientInfo] | None:
    """
    Carica le informazioni base degli ingredienti dal file CSV specificato.

    Funzionalità:
    1. Legge il CSV degli ingredienti (prova prima UTF-8, poi Latin-1 come fallback)
    2. Estrae e converte correttamente i dati nutrizionali e i flag dietetici
    3. Gestisce errori di formattazione, valori mancanti e duplicati

    Args:
        filepath: Percorso completo al file CSV degli ingredienti

    Returns:
        Dizionario {nome_ingrediente: IngredientInfo} con tutti i dati nutrizionali
        e flag dietetici, o None in caso di errore critico (file mancante, colonne obbligatorie assenti)

    Raises:
        Gestisce internamente le eccezioni, stampando messaggi di errore dettagliati
    """
    print(f"--- Caricamento Info Base Ingredienti da {filepath} ---")
    start_time = time.time()
    ingredients_data: Dict[str, IngredientInfo] = {}

    # Funzione helper per il parsing booleano
    def parse_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            val = value.strip().lower()
            if val in ('true', 'vero', '1', 'yes', 'sì', 'si'):
                return True
            if val in ('false', 'falso', '0', 'no'):
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    def safe_float_conversion(value, field_name, ingredient_name):
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            print(
                f"Attenzione (loaders): Valore {field_name} non valido ('{value}') per '{ingredient_name}'. Impostato a None.")
            return None

    try:
        # Verifica esistenza del file
        if not os.path.exists(filepath):
            print(
                f"ERRORE: File CSV degli ingredienti non trovato in: {filepath}")
            print(f"Path corrente: {os.getcwd()}")
            return None

        # Debug per vedere il contenuto del CSV
        print(f"Verifica contenuto CSV: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_lines = ''.join([f.readline() for _ in range(5)])
                print(f"Prime 5 linee del CSV:\n{first_lines}")
        except Exception as e:
            print(f"Errore nella lettura del file CSV per debug: {e}")

        # Usa encoding esplicito per robustezza
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            print(f"CSV caricato con encoding UTF-8, shape: {df.shape}")
        except UnicodeDecodeError:
            print("ATTENZIONE (loaders): Fallita lettura UTF-8, tentativo con Latin-1...")
            df = pd.read_csv(filepath, encoding='latin-1')
            print(f"CSV caricato con encoding Latin-1, shape: {df.shape}")

        # Pulisci nomi colonne
        df.columns = df.columns.str.strip()
        print(f"Colonne disponibili: {df.columns.tolist()}")

        # Verifica colonna 'name' obbligatoria
        if 'name' not in df.columns:
            print(f"ERRORE: Colonna 'name' obbligatoria mancante nel CSV.")
            print(f"Colonne disponibili: {df.columns.tolist()}")
            return None

        # Verifica colonna 'cho_per_100g' obbligatoria
        if 'cho_per_100g' not in df.columns:
            # Tentativo fallback con altri nomi possibili
            if 'carbs' in df.columns:
                df = df.rename(columns={'carbs': 'cho_per_100g'})
                print("Rinominata colonna 'carbs' in 'cho_per_100g'")
            elif 'carbohydrates' in df.columns:
                df = df.rename(columns={'carbohydrates': 'cho_per_100g'})
                print("Rinominata colonna 'carbohydrates' in 'cho_per_100g'")
            else:
                print(f"ERRORE: Colonna 'cho_per_100g' obbligatoria mancante nel CSV.")
                print(f"Colonne disponibili: {df.columns.tolist()}")
                return None

        # Loop per popolare il dizionario
        for index, row in df.iterrows():
            name = str(row['name']).strip()
            if not name:
                # print(f"Attenzione (loaders): Riga {index+2} saltata per nome mancante.")
                continue  # Salta righe senza nome

            # Gestisci CHO (obbligatorio)
            try:
                cho = float(row['cho_per_100g'])
            except (ValueError, TypeError):
                print(
                    f"Attenzione (loaders): Valore CHO non valido ('{row.get('cho_per_100g')}') per '{name}'. Impostato a 0.")
                cho = 0.0

            # Accedi alle altre colonne opzionali usando get con default None
            calories = safe_float_conversion(
                row.get('calories_per_100g'), 'Calorie', name)
            protein = safe_float_conversion(
                row.get('protein_g_per_100g') or row.get('protein_per_100g'), 'Proteine', name)
            fat = safe_float_conversion(
                row.get('fat_g_per_100g') or row.get('fat_per_100g'), 'Grassi', name)
            fiber = safe_float_conversion(
                row.get('fiber_g_per_100g') or row.get('fiber_per_100g'), 'Fibre', name)

            # Gestione flag dietetici con vari nomi possibili di colonna
            is_vegan = parse_bool(row.get('is_vegan', False))
            is_vegetarian = parse_bool(row.get('is_vegetarian', False))

            # Gestisci il fatto che is_gluten_free potrebbe essere una stringa
            is_gluten_free_val = row.get('is_gluten_free', False)
            if isinstance(is_gluten_free_val, str):
                is_gluten_free = parse_bool(is_gluten_free_val)
            else:
                is_gluten_free = bool(is_gluten_free_val)

            is_lactose_free = parse_bool(row.get('is_lactose_free', False))

            ingredient = IngredientInfo(
                name=name,  # Nome originale come chiave
                cho_per_100g=cho,
                calories_per_100g=calories,
                protein_g_per_100g=protein,
                fat_g_per_100g=fat,
                fiber_g_per_100g=fiber,
                # Usa get con default
                is_vegan=is_vegan,
                is_vegetarian=is_vegetarian,
                is_gluten_free=is_gluten_free,
                is_lactose_free=is_lactose_free,
            )

            if name in ingredients_data:
                print(
                    f"Attenzione (loaders): Nome ingrediente duplicato '{name}' alla riga {index+2}. Verrà sovrascritto.")
            ingredients_data[name] = ingredient

        loading_end_time = time.time()
        print(
            f"--- Caricamento Info Base completato ({len(ingredients_data)} ingredienti) in {loading_end_time - start_time:.2f} secondi ---")

        # Debug - mostra alcuni esempi
        sample_items = list(ingredients_data.items())[:3]
        for name, info in sample_items:
            print(
                f"Ingrediente: {name}, CHO: {info.cho_per_100g}, Vegan: {info.is_vegan}")

        return ingredients_data

    except FileNotFoundError:
        print(f"ERRORE (loaders): File ingredienti non trovato a {filepath}")
        return None  # Ritorna None per indicare fallimento critico
    except ValueError as ve:  # Cattura errori di colonna mancante
        print(f"ERRORE (loaders): Problema con le colonne del CSV: {ve}")
        return None
    except Exception as e:
        print(
            f"ERRORE IMPREVISTO (loaders) durante caricamento info base: {e}")
        import traceback
        traceback.print_exc()
        return None
