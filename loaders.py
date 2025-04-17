# loaders.py
import time
from typing import List, Dict, Any, Callable
import pandas as pd
import numpy as np
# SentenceTransformer NON è più necessario qui per caricare i dati base
from utils import normalize_name  # Assumi sia ancora in utils
from model_schema import IngredientInfo

# Rimuovi EMBEDDING_MODEL_NAME se non serve più qui


def load_basic_ingredient_info(filepath: str) -> Dict[str, IngredientInfo] | None:
    """
    Carica solo le informazioni base degli ingredienti (dizionario nome->info) dal CSV.
    NON carica/calcola embedding o modello.
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
        # Usa encoding esplicito per robustezza
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            print("ATTENZIONE (loaders): Fallita lettura UTF-8, tentativo con Latin-1...")
            df = pd.read_csv(filepath, encoding='latin-1')

        # Pulisci nomi colonne
        df.columns = df.columns.str.strip()

        # Verifica colonna 'name' obbligatoria
        if 'name' not in df.columns:
            raise ValueError("Colonna 'name' obbligatoria mancante nel CSV.")
        # Verifica colonna 'cho_per_100g' obbligatoria
        if 'cho_per_100g' not in df.columns:
            raise ValueError(
                "Colonna 'cho_per_100g' obbligatoria mancante nel CSV.")

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
                row.get('protein_g_per_100g'), 'Proteine', name)
            fat = safe_float_conversion(
                row.get('fat_g_per_100g'), 'Grassi', name)
            fiber = safe_float_conversion(
                row.get('fiber_g_per_100g'), 'Fibre', name)

            ingredient = IngredientInfo(
                name=name,  # Nome originale come chiave
                cho_per_100g=cho,
                calories_per_100g=calories,
                protein_g_per_100g=protein,
                fat_g_per_100g=fat,
                fiber_g_per_100g=fiber,
                # Usa get con default
                is_vegan=parse_bool(row.get('is_vegan', False)),
                is_vegetarian=parse_bool(row.get('is_vegetarian', False)),
                is_gluten_free=parse_bool(row.get('is_gluten_free', False)),
                is_lactose_free=parse_bool(row.get('is_lactose_free', False)),
            )

            if name in ingredients_data:
                print(
                    f"Attenzione (loaders): Nome ingrediente duplicato '{name}' alla riga {index+2}. Verrà sovrascritto.")
            ingredients_data[name] = ingredient

        loading_end_time = time.time()
        print(
            f"--- Caricamento Info Base completato ({len(ingredients_data)} ingredienti) in {loading_end_time - start_time:.2f} secondi ---")
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
