
"""
Funzioni di utilità per il sistema di generazione ricette.

Questo modulo contiene funzioni per i calcoli nutrizionali e le verifiche dietetiche delle ricette. Le funzioni qui presenti sono utilizzate da diversi componenti del sistema per eseguire calcoli coerenti di carboidrati e altri valori nutrizionali.
"""
from typing import List, Dict
# Importa le classi da model_schema se necessario per type hinting
from model_schema import RecipeIngredient, IngredientInfo, CalculatedIngredient, Recipe, FinalRecipeOption, UserPreferences


def calculate_total_cho(
    ingredients: List[RecipeIngredient],
    ingredient_data: Dict[str, IngredientInfo]
) -> float:
    """
    Calcola i CHO (carboidrati) totali per una lista di ingredienti di una ricetta.

    Questa funzione itera attraverso ogni ingrediente nella ricetta, cerca le sue informazioni nutrizionali nel dizionario ingredient_data, e calcola il suo contributo di carboidrati in base alla sua quantità in grammi.

    Args:
        ingredients: Lista di oggetti RecipeIngredient dalla ricetta.
        ingredient_data: Dizionario con le informazioni nutrizionali ({nome: IngredientInfo}).

    Returns:
        float: CHO totali calcolati per la ricetta, arrotondati a 2 decimali.

    Note:
        - Se un ingrediente non viene trovato nel dizionario ingredient_data, viene
          mostrato un avviso e l'ingrediente viene ignorato nel calcolo
        - Il calcolo utilizza la formula: (quantità_g * cho_per_100g) / 100.0
        - Se cho_per_100g è None o 0, l'ingrediente non contribuisce ai CHO totali
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
    Calcola il contributo nutrizionale dettagliato di ogni ingrediente in una ricetta.

    Questa funzione è più completa di calculate_total_cho perché calcola non solo
    il contributo in carboidrati, ma anche in calorie, proteine, grassi e fibre
    per ogni ingrediente. Crea una nuova lista di oggetti CalculatedIngredient
    che includono tutte queste informazioni.

    Args:
        ingredients: Lista di oggetti RecipeIngredient dalla ricetta.
        ingredient_data: Dizionario con le informazioni nutrizionali ({nome: IngredientInfo}).

    Returns:
        List[CalculatedIngredient]: Lista di ingredienti con i contributi nutrizionali calcolati.

    Note:
        - Se un ingrediente non viene trovato nel dizionario ingredient_data, viene
          mostrato un avviso e viene restituito un CalculatedIngredient con contributo 0
        - I valori nutrizionali sono calcolati con la formula: (quantità_g * valore_per_100g) / 100.0
        - I valori sono arrotondati a 2 decimali per una migliore leggibilità
        - I valori nutrizionali opzionali (calorie, proteine, grassi, fibre) vengono
          calcolati solo se disponibili nel database degli ingredienti
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
    """
    Verifica se una ricetta soddisfa le preferenze dietetiche dell'utente.

    Questa funzione controlla che i flag dietetici della ricetta (vegano, vegetariano,
    senza glutine, senza lattosio) siano compatibili con le preferenze dell'utente.
    Una ricetta è compatibile se soddisfa tutte le restrizioni dietetiche richieste.

    Args:
        recipe: L'oggetto Recipe da verificare.
        preferences: Le preferenze dell'utente (UserPreferences).

    Returns:
        bool: True se la ricetta soddisfa tutte le preferenze, False altrimenti.

    Note:
        - Se l'utente ha selezionato una preferenza (es. vegano=True), la ricetta
          deve avere il corrispondente flag (is_vegan_recipe=True) per essere compatibile
        - Se l'utente non ha selezionato una preferenza (es. vegano=False), la ricetta
          può avere qualsiasi valore per quel flag
        - Questa funzione è utilizzata nel pre-filtering delle ricette
    """
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
    """
    Verifica se una ricetta finale e verificata soddisfa le preferenze dietetiche dell'utente.

    Funziona in modo simile a check_dietary_match, ma opera su oggetti FinalRecipeOption
    anziché Recipe. Le ricette finali hanno nomi di campi leggermente diversi per i flag
    dietetici rispetto alle ricette iniziali.

    Args:
        recipe: L'oggetto FinalRecipeOption da verificare.
        preferences: Le preferenze dell'utente (UserPreferences).

    Returns:
        bool: True se la ricetta finale soddisfa tutte le preferenze, False altrimenti.

    Note:
        - Questa funzione è utilizzata nella fase finale di verifica delle ricette
        - I nomi dei campi per i flag dietetici sono diversi qui rispetto a check_dietary_match
          (is_vegan invece di is_vegan_recipe, ecc.)
    """
    if preferences.vegan and not recipe.is_vegan:
        return False
    if preferences.vegetarian and not recipe.is_vegetarian:
        return False
    if preferences.gluten_free and not recipe.is_gluten_free:
        return False
    if preferences.lactose_free and not recipe.is_lactose_free:
        return False
    return True
