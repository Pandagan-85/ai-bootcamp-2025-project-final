# utils.py

from typing import List, Dict
# Importa le classi da model_schema se necessario per type hinting
from model_schema import RecipeIngredient, IngredientInfo, CalculatedIngredient, Recipe, FinalRecipeOption, UserPreferences
from langchain_openai import ChatOpenAI
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
    Calcola il contributo nutrizionale di ogni ingrediente in una ricetta.

    Args:
        ingredients: Lista di oggetti RecipeIngredient dalla ricetta.
        ingredient_data: Dizionario con le informazioni nutrizionali ({nome: IngredientInfo}).

    Returns:
        Lista di oggetti CalculatedIngredient con il contributo nutrizionale per ciascuno.
    """
    calculated_list: List[CalculatedIngredient] = []
    for item in ingredients:
        cho_contribution = 0.0
        calories_contribution = None
        protein_contribution_g = None
        fat_contribution_g = None
        fiber_contribution_g = None

        if item.name in ingredient_data:
            info = ingredient_data[item.name]
            # Calcola il contributo CHO
            if info.cho_per_100g is not None:
                cho_contribution = round(
                    (item.quantity_g * info.cho_per_100g) / 100.0, 2)

            # Calcola gli altri contributi nutrizionali se disponibili
            if info.calories_per_100g is not None:
                calories_contribution = round(
                    (item.quantity_g * info.calories_per_100g) / 100.0, 2)

            if info.protein_per_100g is not None:
                protein_contribution_g = round(
                    (item.quantity_g * info.protein_per_100g) / 100.0, 2)

            if info.fat_per_100g is not None:
                fat_contribution_g = round(
                    (item.quantity_g * info.fat_per_100g) / 100.0, 2)

            if info.fiber_per_100g is not None:
                fiber_contribution_g = round(
                    (item.quantity_g * info.fiber_per_100g) / 100.0, 2)
        else:
            print(
                f"Attenzione: Dati per l'ingrediente '{item.name}' non trovati per calcolo contributo nutrizionale.")

        calculated_list.append(
            CalculatedIngredient(
                name=item.name,
                quantity_g=item.quantity_g,
                cho_contribution=cho_contribution,
                calories_contribution=calories_contribution,
                protein_contribution_g=protein_contribution_g,
                fat_contribution_g=fat_contribution_g,
                fiber_contribution_g=fiber_contribution_g
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
