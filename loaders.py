# loaders.py

import pandas as pd
from typing import List, Dict
from model_schema import IngredientInfo, Recipe, RecipeIngredient


def load_ingredients(filepath: str) -> Dict[str, IngredientInfo]:
    """
    Carica il dataset degli ingredienti da un file CSV.
    Assume colonne: name, cho_per_100g, is_vegan, is_vegetarian, is_gluten_free, is_lactose_free
    """
    try:
        df = pd.read_csv(filepath)
        ingredients_dict = {}
        # Assicurati che i nomi delle colonne corrispondano al tuo CSV
        # Converte i booleani da possibili stringhe/numeri a veri booleani
        for _, row in df.iterrows():
            # Semplice conversione, robustezza può essere migliorata
            def parse_bool(value):
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() in ['true', '1', 'yes', 'sì', 'si', 'vero']
                return bool(value)  # Converte 0 a False, altri numeri a True

            ingredient = IngredientInfo(
                name=row['name'].strip(),  # Pulisce spazi extra
                cho_per_100g=float(row['cho_per_100g']),
                is_vegan=parse_bool(row['is_vegan']),
                is_vegetarian=parse_bool(row['is_vegetarian']),
                is_gluten_free=parse_bool(row['is_gluten_free']),
                is_lactose_free=parse_bool(row['is_lactose_free'])
            )
            ingredients_dict[ingredient.name] = ingredient
        print(f"Caricati {len(ingredients_dict)} ingredienti da {filepath}")
        return ingredients_dict
    except FileNotFoundError:
        print(f"Errore: File ingredienti non trovato a {filepath}")
        return {}
    except Exception as e:
        print(f"Errore durante il caricamento degli ingredienti: {e}")
        return {}


def load_recipes(filepath: str) -> List[Recipe]:
    """
    Carica il dataset delle ricette da un file CSV.
    Assume colonne: name, ingredients_json, is_vegan_recipe, is_vegetarian_recipe, is_gluten_free_recipe, is_lactose_free_recipe
    dove ingredients_json è una stringa JSON tipo '[{"name": "Pasta", "quantity_g": 100}, ...]'
    """
    try:
        df = pd.read_csv(filepath)
        recipes_list = []
        import json  # Importa qui per evitare dipendenza globale se non usato altrove

        for _, row in df.iterrows():
            try:
                # Parsa la stringa JSON degli ingredienti
                ingredients_raw = json.loads(row['ingredients_json'])
                recipe_ingredients = [
                    RecipeIngredient(name=ing['name'].strip(
                    ), quantity_g=float(ing['quantity_g']))
                    for ing in ingredients_raw
                ]

                # Converte i booleani (come sopra)
                def parse_bool(value):
                    if isinstance(value, bool):
                        return value
                    if isinstance(value, str):
                        return value.lower() in ['true', '1', 'yes', 'sì', 'si', 'vero']
                    return bool(value)

                recipe = Recipe(
                    name=row['name'].strip(),
                    ingredients=recipe_ingredients,
                    is_vegan_recipe=parse_bool(row['is_vegan_recipe']),
                    is_vegetarian_recipe=parse_bool(
                        row['is_vegetarian_recipe']),
                    is_gluten_free_recipe=parse_bool(
                        row['is_gluten_free_recipe']),
                    is_lactose_free_recipe=parse_bool(
                        row['is_lactose_free_recipe'])
                    # initial_total_cho potrebbe essere calcolato qui se necessario,
                    # ma lo faremo dinamicamente nel flusso per ora
                )
                recipes_list.append(recipe)
            except json.JSONDecodeError:
                print(
                    f"Errore parsing JSON per ricetta: {row.get('name', 'N/A')}")
            except KeyError as ke:
                print(
                    f"Errore chiave mancante ({ke}) per ricetta: {row.get('name', 'N/A')}")
            except ValueError as ve:
                print(
                    f"Errore valore non valido ({ve}) per ricetta: {row.get('name', 'N/A')}")

        print(f"Caricate {len(recipes_list)} ricette da {filepath}")
        return recipes_list
    except FileNotFoundError:
        print(f"Errore: File ricette non trovato a {filepath}")
        return []
    except Exception as e:
        print(f"Errore durante il caricamento delle ricette: {e}")
        return []
