# utils.py

from typing import List, Dict
# Importa le classi da model_schema se necessario per type hinting
from model_schema import RecipeIngredient, IngredientInfo, CalculatedIngredient, Recipe, FinalRecipeOption, UserPreferences
from langchain_community.chat_models import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor


def calculate_total_cho(
    ingredients: List[RecipeIngredient],
    ingredient_data: Dict[str, IngredientInfo]
) -> float:
    """
    Calcola i CHO totali per una lista di ingredienti di una ricetta.

    Args:
        ingredients: Lista di oggetti RecipeIngredient dalla ricetta.
        ingredient_data: Dizionario con le informazioni nutrizionali ({nome: IngredientInfo}).

    Returns:
        CHO totali calcolati per la ricetta.

    Raises:
        KeyError: Se un ingrediente della ricetta non si trova in ingredient_data.
    """
    total_cho = 0.0
    for item in ingredients:
        if item.name in ingredient_data:
            info = ingredient_data[item.name]
            # Assicurati che la divisione per 100 non causi problemi se cho_per_100g è 0
            if info.cho_per_100g is not None and info.cho_per_100g > 0:
                total_cho += (item.quantity_g * info.cho_per_100g) / 100.0
        else:
            # Gestione errore: ingrediente non trovato nel dataset ingredienti
            # Potresti loggare un warning o lanciare un errore più specifico
            print(
                f"Attenzione: Ingrediente '{item.name}' non trovato nel database ingredienti.")
            # O potresti decidere di ignorare questo ingrediente nel calcolo
            # raise KeyError(f"Ingredient data not found for: {item.name}")
    return round(total_cho, 2)  # Arrotonda a 2 decimali


def calculate_ingredient_cho_contribution(
    ingredients: List[RecipeIngredient],
    ingredient_data: Dict[str, IngredientInfo]
) -> List[CalculatedIngredient]:
    """
    Calcola il contributo CHO di ogni ingrediente in una ricetta.

    Args:
        ingredients: Lista di oggetti RecipeIngredient dalla ricetta.
        ingredient_data: Dizionario con le informazioni nutrizionali ({nome: IngredientInfo}).

    Returns:
        Lista di oggetti CalculatedIngredient con il contributo CHO per ciascuno.
    """
    calculated_list: List[CalculatedIngredient] = []
    for item in ingredients:
        cho_contribution = 0.0
        if item.name in ingredient_data:
            info = ingredient_data[item.name]
            if info.cho_per_100g is not None and info.cho_per_100g > 0:
                cho_contribution = round(
                    (item.quantity_g * info.cho_per_100g) / 100.0, 2)
        else:
            print(
                f"Attenzione: Dati per l'ingrediente '{item.name}' non trovati per calcolo contributo CHO.")
            # Potresti assegnare 0 o gestire diversamente

        calculated_list.append(
            CalculatedIngredient(
                name=item.name,
                quantity_g=item.quantity_g,
                cho_contribution=cho_contribution
            )
        )
    return calculated_list


def check_dietary_match(recipe: Recipe, preferences: 'UserPreferences') -> bool:
    """Verifica se una ricetta soddisfa le preferenze dietetiche."""
    if preferences.vegan and not recipe.is_vegan_recipe:
        return False
    if preferences.vegetarian and not recipe.is_vegetarian_recipe:
        return False
    if preferences.gluten_free and not recipe.is_gluten_free_recipe:
        return False
    if preferences.lactose_free and not recipe.is_lactose_free_recipe:
        return False
    return True


def check_final_recipe_dietary_match(recipe: FinalRecipeOption, preferences: 'UserPreferences') -> bool:
    """Verifica se una ricetta FINALE soddisfa le preferenze dietetiche."""
    if preferences.vegan and not recipe.is_vegan:
        return False
    if preferences.vegetarian and not recipe.is_vegetarian:
        return False
    if preferences.gluten_free and not recipe.is_gluten_free:
        return False
    if preferences.lactose_free and not recipe.is_lactose_free:
        return False
    return True

# Potresti aggiungere altre funzioni utili qui, es. per validare l'equilibrio
# (anche se per l'MVP ci affidiamo al prompt dell'LLM)
