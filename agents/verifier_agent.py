"""
Agente di verifica e ottimizzazione delle ricette generate.

Questo modulo implementa l'agente verificatore potenziato, responsabile dell'analisi,
matching, ottimizzazione e verifica delle ricette generate. Questo agente è il
"cervello" del sistema in grado di correggere e migliorare le ricette per soddisfare
i requisiti nutrizionali e dietetici.
"""
from typing import List, Dict, Optional, Tuple
from copy import deepcopy
import random

from model_schema import GraphState, FinalRecipeOption, UserPreferences, RecipeIngredient, IngredientInfo, CalculatedIngredient
from utils import find_best_match_faiss, calculate_ingredient_cho_contribution, normalize_name

# --- FUNZIONI DI OTTIMIZZAZIONE ---


def calculate_recipe_similarity(recipe1: FinalRecipeOption, recipe2: FinalRecipeOption) -> float:
    """
    Calcola un punteggio di somiglianza tra due ricette basato su vari fattori.

    Criteri di similarità (con pesi differenti):
    1. Somiglianza nel titolo (peso: 0.2) - escluse parole comuni
    2. Ingredienti principali condivisi (peso: 0.4) - top 3 per quantità
    3. Tipo di piatto basato su parole chiave (peso: 0.25)
    4. Attributi dietetici comuni (peso: 0.15) - vegano, vegetariano, ecc.

    Args:
        recipe1, recipe2: Le ricette da confrontare

    Returns:
        Punteggio da 0.0 (completamente diverse) a 1.0 (identiche)
    """
    similarity_score = 0.0
    total_weight = 0.0

    # 1. Somiglianza nel titolo (peso: 0.2)
    weight = 0.2
    title1_words = set(recipe1.name.lower().split())
    title2_words = set(recipe2.name.lower().split())
    # Rimuovi parole comuni
    common_words = {"con", "e", "al", "di", "la", "il",
                    "le", "i", "in", "del", "della", "allo", "alla"}
    title1_words = title1_words - common_words
    title2_words = title2_words - common_words

    if title1_words and title2_words:  # Evita divisione per zero
        title_overlap = len(title1_words.intersection(
            title2_words)) / min(len(title1_words), len(title2_words))
        similarity_score += title_overlap * weight
        total_weight += weight

    # 2. Ingredienti principali (peso: 0.4)
    weight = 0.4
    # Estrai gli ingredienti principali (top 3 per grammi)

    def get_main_ingredients(recipe):
        sorted_ingredients = sorted(
            recipe.ingredients, key=lambda x: x.quantity_g, reverse=True)
        return {ing.name for ing in sorted_ingredients[:3]}

    main_ingredients1 = get_main_ingredients(recipe1)
    main_ingredients2 = get_main_ingredients(recipe2)

    if main_ingredients1 and main_ingredients2:
        ingredients_overlap = len(main_ingredients1.intersection(
            main_ingredients2)) / min(len(main_ingredients1), len(main_ingredients2))
        similarity_score += ingredients_overlap * weight
        total_weight += weight

    # 3. Tipo di piatto basato su parole chiave (peso: 0.25)
    weight = 0.25
    dish_categories = {
        "primo": {"pasta", "risotto", "zuppa", "minestra", "minestrone", "gnocchi", "spaghetti", "lasagne", "riso"},
        "secondo": {"pollo", "manzo", "tacchino", "vitello", "bistecca", "pesce", "salmone", "tonno", "frittata", "uova", "polpette"},
        "contorno": {"insalata", "verdure", "vegetali", "patate", "legumi"},
        "dessert": {"torta", "dolce", "gelato", "budino", "crema", "crostata"}
    }

    def get_dish_type(recipe: FinalRecipeOption):  # Accetta l'intero oggetto

        name_lower = recipe.name.lower()
        for category, keywords in dish_categories.items():
            for keyword in keywords:
                if keyword in name_lower:
                    return category
        # Controlla anche gli ingredienti
        ingredients_text = " ".join([ing.name.lower()
                                    for ing in recipe.ingredients])
        for category, keywords in dish_categories.items():
            for keyword in keywords:
                if keyword in ingredients_text:
                    return category
        return "unknown"

    dish_type1 = get_dish_type(recipe1)
    dish_type2 = get_dish_type(recipe2)

    if dish_type1 == dish_type2 and dish_type1 != "unknown":
        similarity_score += weight
        total_weight += weight

    # 4. Attributi dietetici (peso: 0.15)
    weight = 0.15
    dietary_attrs1 = (recipe1.is_vegan, recipe1.is_vegetarian,
                      recipe1.is_gluten_free, recipe1.is_lactose_free)
    dietary_attrs2 = (recipe2.is_vegan, recipe2.is_vegetarian,
                      recipe2.is_gluten_free, recipe2.is_lactose_free)

    dietary_similarity = sum(a == b for a, b in zip(
        dietary_attrs1, dietary_attrs2)) / 4.0
    similarity_score += dietary_similarity * weight
    total_weight += weight

    # Normalizza il punteggio totale
    return similarity_score / total_weight if total_weight > 0 else 0.0


def ensure_recipe_diversity(recipes: List[FinalRecipeOption], target_cho: float, similarity_threshold: float = 0.6) -> List[FinalRecipeOption]:
    """
    Filtra una lista di ricette per assicurarsi che non ci siano ricette troppo simili tra loro.

    Implementazione:
    1. Ordina le ricette per qualità (vicinanza al target CHO)
    2. Parte dalla ricetta migliore e aggiunge solo ricette sufficientemente diverse

    Args:
        recipes: Lista di ricette da filtrare
        target_cho: Valore target di carboidrati per la valutazione della qualità
        similarity_threshold: Soglia sopra la quale le ricette sono considerate troppo simili

    Returns:
        Lista di ricette filtrata per massimizzare diversità mantenendo qualità
    """
    if len(recipes) <= 1:
        return recipes

    # Ordina ricette per qualità (in base alla distanza dal target CHO)
    sorted_recipes = sorted(recipes, key=lambda r: abs(
        r.total_cho - target_cho) if r.total_cho else float('inf'))

    # Lista per le ricette diverse
    diverse_recipes = [sorted_recipes[0]]  # Inizia con la migliore ricetta

    # Controlla le ricette rimanenti
    for candidate in sorted_recipes[1:]:
        # Calcola similarità con tutte le ricette già selezionate
        is_too_similar = False
        for selected in diverse_recipes:
            similarity = calculate_recipe_similarity(candidate, selected)
            if similarity > similarity_threshold:
                is_too_similar = True
                print(
                    f"Ricetta '{candidate.name}' scartata: troppo simile a '{selected.name}' (similarità: {similarity:.2f})")
                break

        if not is_too_similar:
            diverse_recipes.append(candidate)

    return diverse_recipes


def correct_dietary_flags(recipe: FinalRecipeOption, ingredient_data: Dict[str, IngredientInfo]) -> FinalRecipeOption:
    """
    Corregge i flag dietetici di una ricetta basandosi sugli ingredienti contenuti.

    Utilizza liste predefinite di ingredienti non compatibili con ciascuna categoria
    dietetica per verificare e correggere i flag, esaminando gli ingredienti
    della ricetta.

    Args:
        recipe: Ricetta da verificare
        ingredient_data: Database degli ingredienti con informazioni nutrizionali

    Returns:
        Ricetta con flag dietetici corretti in base agli ingredienti effettivi
    """
    # Lista di ingredienti NON vegani
    non_vegan_ingredients = {"pollo", "tacchino", "manzo", "vitello", "maiale", "prosciutto",
                             "pancetta", "salmone", "tonno", "pesce", "uova", "uovo", "formaggio",
                             "parmigiano", "mozzarella", "ricotta", "burro", "latte", "panna"}

    # Lista di ingredienti NON vegetariani
    non_vegetarian_ingredients = {"pollo", "tacchino", "manzo", "vitello", "maiale", "prosciutto",
                                  "pancetta", "salmone", "tonno", "pesce"}

    # Lista di ingredienti NON senza glutine
    gluten_ingredients = {"pasta", "pane", "farina", "couscous", "orzo", "farro",
                          "seitan", "pangrattato", "grano"}

    # Lista di ingredienti NON senza lattosio
    lactose_ingredients = {"latte", "formaggio", "parmigiano", "mozzarella", "ricotta",
                           "burro", "panna", "yogurt"}

    updated_recipe = deepcopy(recipe)

    # Controlla ogni ingrediente
    ing_names_lower = [ing.name.lower() for ing in recipe.ingredients]
    combined_text = " ".join(ing_names_lower).lower()

    # Check vegano
    for item in non_vegan_ingredients:
        if item in combined_text:
            updated_recipe.is_vegan = False
            break

    # Check vegetariano
    for item in non_vegetarian_ingredients:
        if item in combined_text:
            updated_recipe.is_vegetarian = False
            break

    # Check senza glutine
    for item in gluten_ingredients:
        if item in combined_text:
            updated_recipe.is_gluten_free = False
            break

    # Check senza lattosio
    for item in lactose_ingredients:
        if item in combined_text:
            updated_recipe.is_lactose_free = False
            break

    return updated_recipe


def identify_cho_contributors(recipe: FinalRecipeOption, ingredient_data: Dict[str, IngredientInfo]) -> List[CalculatedIngredient]:
    """
    Identifica gli ingredienti che contribuiscono maggiormente ai carboidrati (CHO) nella ricetta.

    Args:
        recipe: Ricetta da analizzare
        ingredient_data: Database ingredienti con informazioni nutrizionali

    Returns:
        Lista di ingredienti ordinati per contributo CHO (dal maggiore al minore)
        Include sia ingredienti con contributo CHO già calcolato che ingredienti
        identificati come ricchi di CHO nel database anche se non ancora calcolati
    """
    # Filtra solo ingredienti con contributo CHO significativo
    cho_rich_ingredients = []

    for ing in recipe.ingredients:
        # Controlla se l'ingrediente ha un contributo CHO
        if hasattr(ing, 'cho_contribution') and ing.cho_contribution is not None and ing.cho_contribution > 0:
            cho_rich_ingredients.append(ing)
        # Gestisci gli ingredienti che hanno nome ma non contributo calcolato
        elif ing.name in ingredient_data and ingredient_data[ing.name].cho_per_100g > 5:
            # Ingrediente ricco di CHO nel DB ma non calcolato nella ricetta
            cho_rich_ingredients.append(ing)

    # Ordina per contributo CHO (se disponibile) o per CHO/100g dal DB
    cho_rich_ingredients.sort(
        key=lambda x: (x.cho_contribution if hasattr(x, 'cho_contribution') and x.cho_contribution is not None else
                       (ingredient_data[x.name].cho_per_100g * x.quantity_g / 100 if x.name in ingredient_data else 0)),
        reverse=True
    )

    return cho_rich_ingredients


def fine_tune_recipe(recipe: FinalRecipeOption, ingredient_to_adjust: CalculatedIngredient,
                     cho_difference: float, ingredient_data: Dict[str, IngredientInfo]) -> FinalRecipeOption:
    """
    Effettua un aggiustamento fine della ricetta modificando un singolo ingrediente.

    Strategia utilizzata per piccole differenze di CHO (< 15g) rispetto al target.
    Modifica la quantità di un ingrediente specifico per compensare la differenza di CHO.

    Args:
        recipe: Ricetta da aggiustare
        ingredient_to_adjust: Ingrediente da modificare
        cho_difference: Differenza di CHO da compensare (target - attuale)
        ingredient_data: Database ingredienti con informazioni nutrizionali

    Returns:
        Ricetta modificata con valori nutrizionali ricalcolati
    """
    adjusted_recipe = deepcopy(recipe)

    # Trova l'ingrediente da modificare nella ricetta
    for i, ing in enumerate(adjusted_recipe.ingredients):
        if ing.name == ingredient_to_adjust.name:
            # Calcola la nuova quantità basata sul contenuto CHO/100g
            if ing.name in ingredient_data and ingredient_data[ing.name].cho_per_100g > 0:
                cho_per_100g = ingredient_data[ing.name].cho_per_100g
                # Calcola quanti grammi aggiungere/togliere
                gram_change = (cho_difference / cho_per_100g) * 100
                original_quantity = ing.quantity_g
                new_quantity = max(5, original_quantity +
                                   gram_change)  # Minimo 5g

                print(f"Aggiustamento fine: '{ing.name}' da {original_quantity:.1f}g a {new_quantity:.1f}g " +
                      f"(Cambio: {gram_change:+.1f}g per compensare {cho_difference:+.1f}g CHO)")

                # Aggiorna la quantità
                adjusted_recipe.ingredients[i].quantity_g = round(
                    new_quantity, 1)
                break

    # Ricalcola i valori nutrizionali
    updated_ingredients = calculate_ingredient_cho_contribution(
        adjusted_recipe.ingredients, ingredient_data
    )
    adjusted_recipe.ingredients = updated_ingredients

    # Aggiorna i totali
    adjusted_recipe.total_cho = sum(
        ing.cho_contribution for ing in updated_ingredients if ing.cho_contribution is not None)
    adjusted_recipe.total_calories = sum(
        ing.calories_contribution for ing in updated_ingredients if ing.calories_contribution is not None)
    adjusted_recipe.total_protein_g = sum(
        ing.protein_contribution_g for ing in updated_ingredients if ing.protein_contribution_g is not None)
    adjusted_recipe.total_fat_g = sum(
        ing.fat_contribution_g for ing in updated_ingredients if ing.fat_contribution_g is not None)
    adjusted_recipe.total_fiber_g = sum(
        ing.fiber_contribution_g for ing in updated_ingredients if ing.fiber_contribution_g is not None)

    # Aggiorna il nome se modificato significativamente
    if abs(cho_difference) > 5:
        adjusted_recipe.name = f"{recipe.name} (Ottimizzata)"

    return adjusted_recipe


def adjust_recipe_proportionally(recipe: FinalRecipeOption, cho_contributors: List[CalculatedIngredient],
                                 scaling_factor: float, ingredient_data: Dict[str, IngredientInfo]) -> FinalRecipeOption:
    """
    Aggiusta proporzionalmente tutti gli ingredienti ricchi di CHO in una ricetta.

    Strategia utilizzata per differenze di CHO più significative (> 15g) rispetto al target.
    Applica un fattore di scala a tutti gli ingredienti che contribuiscono ai CHO.

    Args:
        recipe: Ricetta da aggiustare
        cho_contributors: Lista di ingredienti ricchi di CHO
        scaling_factor: Fattore di scala da applicare (es: 1.2 per aumentare del 20%)
        ingredient_data: Database ingredienti con informazioni nutrizionali

    Returns:
        Ricetta modificata con valori nutrizionali ricalcolati
    """
    adjusted_recipe = deepcopy(recipe)
    contributor_names = [ing.name for ing in cho_contributors]

    # Applica scaling a tutti gli ingredienti CHO
    for i, ing in enumerate(adjusted_recipe.ingredients):
        if ing.name in contributor_names:
            original_quantity = ing.quantity_g
            # Applica scaling con limiti
            new_quantity = max(5, min(250, original_quantity * scaling_factor))
            adjusted_recipe.ingredients[i].quantity_g = round(new_quantity, 1)

            print(
                f"Scaling: '{ing.name}' da {original_quantity:.1f}g a {new_quantity:.1f}g (Fattore: {scaling_factor:.2f})")

    # Ricalcola i valori nutrizionali
    updated_ingredients = calculate_ingredient_cho_contribution(
        adjusted_recipe.ingredients, ingredient_data
    )
    adjusted_recipe.ingredients = updated_ingredients

    # Aggiorna i totali
    adjusted_recipe.total_cho = sum(
        ing.cho_contribution for ing in updated_ingredients if ing.cho_contribution is not None)
    adjusted_recipe.total_calories = sum(
        ing.calories_contribution for ing in updated_ingredients if ing.calories_contribution is not None)
    adjusted_recipe.total_protein_g = sum(
        ing.protein_contribution_g for ing in updated_ingredients if ing.protein_contribution_g is not None)
    adjusted_recipe.total_fat_g = sum(
        ing.fat_contribution_g for ing in updated_ingredients if ing.fat_contribution_g is not None)
    adjusted_recipe.total_fiber_g = sum(
        ing.fiber_contribution_g for ing in updated_ingredients if ing.fiber_contribution_g is not None)

    # Aggiorna il nome
    adjusted_recipe.name = f"{recipe.name} (Ottimizzata)"

    return adjusted_recipe


def classify_ingredients_by_cho_contribution(recipe: FinalRecipeOption) -> Dict[str, List[CalculatedIngredient]]:
    """
    Classifica gli ingredienti per contributo CHO relativo alla ricetta.

    Args:
        recipe: Ricetta da analizzare

    Returns:
        Dizionario con ingredienti classificati per categoria (primary, secondary, minor, non_cho)
    """
    total_cho = recipe.total_cho if recipe.total_cho else 0
    classified = {'primary': [], 'secondary': [], 'minor': [], 'non_cho': []}

    if total_cho == 0:
        return classified

    for ing in recipe.ingredients:
        if not hasattr(ing, 'cho_contribution') or ing.cho_contribution is None:
            classified['non_cho'].append(ing)
            continue

        cho_percent = (ing.cho_contribution / total_cho) * 100

        if cho_percent > 50:
            classified['primary'].append(ing)
        elif cho_percent > 10:
            classified['secondary'].append(ing)
        elif cho_percent > 0:
            classified['minor'].append(ing)
        else:
            classified['non_cho'].append(ing)

    # Ordina per contributo decrescente in ogni categoria
    for category in ['primary', 'secondary', 'minor']:
        classified[category].sort(
            key=lambda x: x.cho_contribution, reverse=True)

    return classified


def scale_ingredients_cascade(
    recipe: FinalRecipeOption,
    classified_ingredients: Dict[str, List[CalculatedIngredient]],
    target_cho: float,
    ingredient_data: Dict[str, IngredientInfo]
) -> Optional[FinalRecipeOption]:
    """
    Ottimizza la ricetta usando una strategia a cascata.

    Args:
        recipe: Ricetta originale
        classified_ingredients: Ingredienti classificati per contributo CHO
        target_cho: Target CHO in grammi
        ingredient_data: Database ingredienti

    Returns:
        Ricetta ottimizzata o None se non possibile
    """
    updated_recipe = deepcopy(recipe)
    current_cho = recipe.total_cho if recipe.total_cho else 0
    cho_difference = target_cho - current_cho

    # FASE 1: Prova a ottimizzare SOLO la fonte primaria (se esiste)
    if classified_ingredients['primary']:
        primary_ing = classified_ingredients['primary'][0]
        primary_cho = primary_ing.cho_contribution

        # Calcola il fattore di scala necessario per la fonte primaria
        if primary_cho > 0:
            needed_scale = 1 + (cho_difference / primary_cho)

            # Limita lo scaling per mantenere proporzioni realistiche
            if cho_difference > 0:  # Aumentare CHO
                max_scale = 1.5  # Max +50%
                actual_scale = min(needed_scale, max_scale)
            else:  # Ridurre CHO
                min_scale = 0.5  # Max -50%
                actual_scale = max(needed_scale, min_scale)

            # Applica lo scaling alla fonte primaria
            for i, ing in enumerate(updated_recipe.ingredients):
                if ing.name == primary_ing.name:
                    original_qty = ing.quantity_g
                    # Limiti assoluti
                    new_qty = max(5, min(300, original_qty * actual_scale))
                    updated_recipe.ingredients[i].quantity_g = round(
                        new_qty, 1)
                    print(
                        f"FASE 1 - Scaling primario: '{ing.name}' da {original_qty:.1f}g a {new_qty:.1f}g")
                    break

            # Ricalcola CHO
            updated_recipe.ingredients = calculate_ingredient_cho_contribution(
                updated_recipe.ingredients, ingredient_data)
            updated_recipe.total_cho = sum(
                ing.cho_contribution for ing in updated_recipe.ingredients if ing.cho_contribution is not None)

            # Se siamo nel range accettabile, ritorna
            if abs(updated_recipe.total_cho - target_cho) <= 5:
                return updated_recipe

    # FASE 2: Se necessario, aggiungi scaling limitato delle fonti secondarie
    remaining_difference = target_cho - updated_recipe.total_cho
    if abs(remaining_difference) > 5 and classified_ingredients['secondary']:
        total_secondary_cho = sum(
            ing.cho_contribution for ing in classified_ingredients['secondary'])

        if total_secondary_cho > 0:
            secondary_scale = 1 + (remaining_difference / total_secondary_cho)
            # Limita a ±30% per gli ingredienti secondari
            secondary_scale = max(0.7, min(1.3, secondary_scale))

            for ing in classified_ingredients['secondary']:
                for i, recipe_ing in enumerate(updated_recipe.ingredients):
                    if recipe_ing.name == ing.name:
                        original_qty = recipe_ing.quantity_g
                        new_qty = max(
                            5, min(200, original_qty * secondary_scale))
                        updated_recipe.ingredients[i].quantity_g = round(
                            new_qty, 1)
                        print(
                            f"FASE 2 - Scaling secondario: '{ing.name}' da {original_qty:.1f}g a {new_qty:.1f}g")
                        break

            # Ricalcola CHO
            updated_recipe.ingredients = calculate_ingredient_cho_contribution(
                updated_recipe.ingredients, ingredient_data)
            updated_recipe.total_cho = sum(
                ing.cho_contribution for ing in updated_recipe.ingredients if ing.cho_contribution is not None)

            # Se siamo nel range accettabile, ritorna
            if abs(updated_recipe.total_cho - target_cho) <= 8:
                return updated_recipe

    # FASE 3: Solo se assolutamente necessario, considera minimi aggiustamenti alle fonti minori
    remaining_difference = target_cho - updated_recipe.total_cho
    if abs(remaining_difference) > 8 and classified_ingredients['minor']:
        total_minor_cho = sum(
            ing.cho_contribution for ing in classified_ingredients['minor'])

        if total_minor_cho > 0:
            minor_scale = 1 + (remaining_difference / total_minor_cho)
            # Limita a ±20% per gli ingredienti minori
            minor_scale = max(0.8, min(1.2, minor_scale))

            for ing in classified_ingredients['minor']:
                for i, recipe_ing in enumerate(updated_recipe.ingredients):
                    if recipe_ing.name == ing.name:
                        original_qty = recipe_ing.quantity_g
                        new_qty = max(5, min(150, original_qty * minor_scale))
                        updated_recipe.ingredients[i].quantity_g = round(
                            new_qty, 1)
                        print(
                            f"FASE 3 - Scaling minore: '{ing.name}' da {original_qty:.1f}g a {new_qty:.1f}g")
                        break

            # Ricalcola CHO finale
            updated_recipe.ingredients = calculate_ingredient_cho_contribution(
                updated_recipe.ingredients, ingredient_data)
            updated_recipe.total_cho = sum(
                ing.cho_contribution for ing in updated_recipe.ingredients if ing.cho_contribution is not None)

    # Aggiorna anche gli altri valori nutrizionali
    updated_recipe.total_calories = sum(
        ing.calories_contribution for ing in updated_recipe.ingredients if ing.calories_contribution is not None)
    updated_recipe.total_protein_g = sum(
        ing.protein_contribution_g for ing in updated_recipe.ingredients if ing.protein_contribution_g is not None)
    updated_recipe.total_fat_g = sum(
        ing.fat_contribution_g for ing in updated_recipe.ingredients if ing.fat_contribution_g is not None)
    updated_recipe.total_fiber_g = sum(
        ing.fiber_contribution_g for ing in updated_recipe.ingredients if ing.fiber_contribution_g is not None)

    # Aggiorna il nome se modificato significativamente
    if abs(cho_difference) > 10:
        updated_recipe.name = f"{recipe.name} (Ottimizzata)"

    return updated_recipe


# SOSTITUISCI la funzione optimize_recipe_cho esistente con questa:
def optimize_recipe_cho(recipe: FinalRecipeOption, target_cho: float, ingredient_data: Dict[str, IngredientInfo]) -> Optional[FinalRecipeOption]:
    """
    Ottimizza una ricetta per raggiungere il target CHO utilizzando una strategia a cascata.

    Processo:
    1. Se la ricetta è già nel range target (±5g), la mantiene inalterata
    2. Classifica gli ingredienti per contributo CHO (primario, secondario, minore)
    3. Prova prima a ottimizzare solo la fonte primaria
    4. Se necessario, aggiusta anche le fonti secondarie (con limiti più stretti)
    5. Solo come ultima risorsa, modifica leggermente le fonti minori

    Args:
        recipe: Ricetta da ottimizzare
        target_cho: Target CHO in grammi
        ingredient_data: Database ingredienti con informazioni nutrizionali

    Returns:
        Ricetta ottimizzata o None se non è stato possibile ottimizzarla
    """
    # 1. Calcola i valori nutrizionali attuali se non già calcolati
    if recipe.total_cho is None:
        updated_ingredients = calculate_ingredient_cho_contribution(
            recipe.ingredients, ingredient_data
        )
        recipe.ingredients = updated_ingredients
        recipe.total_cho = sum(
            ing.cho_contribution for ing in updated_ingredients if ing.cho_contribution is not None)

    # Se già nel range target (±5g), non fare nulla
    if abs(recipe.total_cho - target_cho) < 5:
        return recipe

    current_cho = recipe.total_cho
    cho_difference = target_cho - current_cho
    print(
        f"Ottimizzazione: '{recipe.name}' - CHO attuale: {current_cho:.1f}g, Target: {target_cho:.1f}g, Diff: {cho_difference:+.1f}g")

    # 2. Classifica gli ingredienti per contributo CHO
    classified_ingredients = classify_ingredients_by_cho_contribution(recipe)

    # Stampa classificazione per debug
    print("Classificazione ingredienti:")
    for category, ingredients in classified_ingredients.items():
        print(
            f"{category}: {[f'{ing.name} ({ing.cho_contribution:.1f}g)' for ing in ingredients]}")

    # 3. Verifica se ci sono ingredienti che possono essere ottimizzati
    if not classified_ingredients['primary'] and not classified_ingredients['secondary'] and not classified_ingredients['minor']:
        print(
            f"Ottimizzazione fallita: Nessun ingrediente CHO trovato in '{recipe.name}'")
        return None

    # 4. Procedi con l'ottimizzazione a cascata
    optimized_recipe = scale_ingredients_cascade(
        recipe, classified_ingredients, target_cho, ingredient_data)

    # 5. Verifica se l'ottimizzazione ha migliorato la situazione
    if optimized_recipe is None or (abs(optimized_recipe.total_cho - target_cho) >= abs(current_cho - target_cho)):
        print(f"Ottimizzazione fallita: Non è stato possibile avvicinare la ricetta al target CHO")
        return None

    return optimized_recipe


def match_recipe_ingredients(recipe: FinalRecipeOption,
                             ingredient_data: Dict[str, IngredientInfo],
                             normalized_to_original: Dict[str, str],
                             original_to_normalized: Dict[str, str],
                             faiss_index,
                             index_to_name_mapping,
                             embedding_model,
                             normalize_function) -> Tuple[FinalRecipeOption, bool]:
    """
    Effettua il matching degli ingredienti della ricetta con il database usando FAISS.
    Utilizza mappature normalizzate per un matching coerente.
    """
    matched_recipe = deepcopy(recipe)
    matched_ingredients = []
    all_matched = True

    print(f"Matching ingredienti per ricetta '{recipe.name}'")

    # DEBUG - Verifica la presenza dei dati nel database
    print(
        f"DEBUG: Database ingredienti contiene {len(ingredient_data)} elementi")
    print(
        f"DEBUG: Mapping normalizzato contiene {len(normalized_to_original)} elementi")
    sample_keys = list(ingredient_data.keys())[:5]
    print(f"DEBUG: Primi 5 ingredienti nel DB: {sample_keys}")

    # Dizionario per le corrispondenze comuni per ingredienti problematici
    common_fallbacks = {
        "pomodori": "Pomodoro",  # Nome esatto nel CSV
        "pomodoro fresco": "Pomodoro",
        "pomodori freschi": "Pomodoro",
        "pomodoro a cubetti": "Pomodoro",
        "pomodori a cubetti": "Pomodoro",
        "peperoni": "Peperoni",
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
        "origano fresco": "Origano",
        "origano secco": "Origano",
        "mirtilli rossi": "Mirtilli",
        "prezzemolo fresco": "Prezzemolo",
        "zenzero fresco": "Zenzero",
        "zenzero fresco grattugiato": "Zenzero",
        "limoni": "Limone",  # Chiave: variante norm. Valore: nome esatto CSV
        "scorza di limone": "Limone",  # Mappiamo anche la scorza al limone generico
        "scorza di limone grattugiata": "Limone",
        "lievi scorze di limone grattugiate": "Limone",
    }

    for ing in recipe.ingredients:
        # Tenta il matching con FAISS
        match_result = find_best_match_faiss(
            llm_name=ing.name,
            faiss_index=faiss_index,
            index_to_name_mapping=index_to_name_mapping,
            model=embedding_model,
            normalize_func=normalize_function,
            threshold=0.60
        )

        ingredient_matched = False

        if match_result:
            matched_db_name, match_score = match_result
            print(
                f"Ingrediente '{ing.name}' matchato a '{matched_db_name}' (score: {match_score:.2f})")

            # SEQUENZA DI TENTATIVI DI MATCH

            # 1. Prova con il nome esatto restituito dal matching
            if matched_db_name in ingredient_data:
                print(
                    f"Trovato direttamente con nome matchato: '{matched_db_name}'")
                matched_ingredients.append(
                    RecipeIngredient(name=matched_db_name,
                                     quantity_g=ing.quantity_g)
                )
                ingredient_matched = True
                continue

            # 2. Prova con la normalizzazione e mappatura
            normalized_name = normalize_function(matched_db_name)
            if normalized_name in normalized_to_original:
                original_db_name = normalized_to_original[normalized_name]
                print(
                    f"Usando mappatura normalizzata: '{matched_db_name}' -> '{original_db_name}'")

                if original_db_name in ingredient_data:
                    matched_ingredients.append(
                        RecipeIngredient(name=original_db_name,
                                         quantity_g=ing.quantity_g)
                    )
                    ingredient_matched = True
                    continue

            # 3. Tenta una ricerca case-insensitive nel database
            found = False
            for db_ingredient in ingredient_data.keys():
                if matched_db_name.lower() == db_ingredient.lower():
                    print(
                        f"Match case-insensitive: '{matched_db_name}' -> '{db_ingredient}'")
                    matched_ingredients.append(
                        RecipeIngredient(name=db_ingredient,
                                         quantity_g=ing.quantity_g)
                    )
                    found = True
                    ingredient_matched = True
                    break

            if found:
                continue

            # 4. Prova fallback per ingredienti problematici
            normalized_matched = normalize_function(matched_db_name)
            if normalized_matched in common_fallbacks:
                fallback_name = common_fallbacks[normalized_matched]
                if fallback_name in ingredient_data:
                    print(
                        f"Usando fallback: '{matched_db_name}' -> '{fallback_name}'")
                    matched_ingredients.append(
                        RecipeIngredient(name=fallback_name,
                                         quantity_g=ing.quantity_g)
                    )
                    ingredient_matched = True
                    continue

            # 5. Se ancora non trovato, prova a cercare con il nome originale dell'LLM
            if ing.name in ingredient_data:
                print(f"Usando nome LLM originale: '{ing.name}'")
                matched_ingredients.append(
                    RecipeIngredient(name=ing.name, quantity_g=ing.quantity_g)
                )
                ingredient_matched = True
                continue

            # Nome originale normalizzato
            normalized_original = normalize_function(ing.name)
            if normalized_original in common_fallbacks:
                fallback_name = common_fallbacks[normalized_original]
                if fallback_name in ingredient_data:
                    print(
                        f"Usando fallback da originale: '{ing.name}' -> '{fallback_name}'")
                    matched_ingredients.append(
                        RecipeIngredient(name=fallback_name,
                                         quantity_g=ing.quantity_g)
                    )
                    ingredient_matched = True
                    continue
        else:
            print(f"Nessun match trovato per '{ing.name}'")

            # Prova fallback per ingredienti non matchati
            normalized_original = normalize_function(ing.name)
            if normalized_original in common_fallbacks:
                fallback_name = common_fallbacks[normalized_original]
                if fallback_name in ingredient_data:
                    print(
                        f"Usando fallback per non matchato: '{ing.name}' -> '{fallback_name}'")
                    matched_ingredients.append(
                        RecipeIngredient(name=fallback_name,
                                         quantity_g=ing.quantity_g)
                    )
                    ingredient_matched = True
                    continue

        # Se arriviamo qui, nessuno dei tentativi ha avuto successo
        if not ingredient_matched:
            all_matched = False
            print(
                f"ERRORE: '{matched_db_name if match_result else ing.name}' non trovato nel database degli ingredienti!")
            matched_ingredients.append(
                RecipeIngredient(
                    name=f"{ing.name} (Info Mancanti!)", quantity_g=ing.quantity_g)
            )

    # Calcola valori nutrizionali
    calculated_ingredients = calculate_ingredient_cho_contribution(
        matched_ingredients, ingredient_data
    )

    # Aggiorna ricetta
    matched_recipe.ingredients = calculated_ingredients

    # Calcola totali solo se tutti gli ingredienti sono stati matchati
    if all_matched:
        matched_recipe.total_cho = sum(
            ing.cho_contribution for ing in calculated_ingredients if ing.cho_contribution is not None)
        matched_recipe.total_calories = sum(
            ing.calories_contribution for ing in calculated_ingredients if ing.calories_contribution is not None)
        matched_recipe.total_protein_g = sum(
            ing.protein_contribution_g for ing in calculated_ingredients if ing.protein_contribution_g is not None)
        matched_recipe.total_fat_g = sum(
            ing.fat_contribution_g for ing in calculated_ingredients if ing.fat_contribution_g is not None)
        matched_recipe.total_fiber_g = sum(
            ing.fiber_contribution_g for ing in calculated_ingredients if ing.fiber_contribution_g is not None)

    return matched_recipe, all_matched


def verify_dietary_preferences(recipe: FinalRecipeOption, preferences: UserPreferences) -> bool:
    """
    Verifica che la ricetta soddisfi le preferenze dietetiche dell'utente.

    Args:
        recipe: Ricetta da verificare
        preferences: Preferenze dell'utente

    Returns:
        True se la ricetta soddisfa le preferenze, False altrimenti
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


def compute_dietary_flags(recipe: FinalRecipeOption, ingredient_data: Dict[str, IngredientInfo]) -> FinalRecipeOption:
    """
    Calcola i flag dietetici (vegan, vegetarian, ecc.) in base agli ingredienti.

    Args:
        recipe: Ricetta da analizzare
        ingredient_data: Database ingredienti

    Returns:
        Ricetta con flag dietetici aggiornati
    """
    updated_recipe = deepcopy(recipe)

    # Default a True, diventerà False se trovato ingrediente non compatibile
    is_vegan = True
    is_vegetarian = True
    is_gluten_free = True
    is_lactose_free = True

    for ing in recipe.ingredients:
        if ing.name in ingredient_data:
            info = ingredient_data[ing.name]
            if not info.is_vegan:
                is_vegan = False
            if not info.is_vegetarian:
                is_vegetarian = False
            if not info.is_gluten_free:
                is_gluten_free = False
            if not info.is_lactose_free:
                is_lactose_free = False

    updated_recipe.is_vegan = is_vegan
    updated_recipe.is_vegetarian = is_vegetarian
    updated_recipe.is_gluten_free = is_gluten_free
    updated_recipe.is_lactose_free = is_lactose_free

    return updated_recipe


def add_ingredient(recipe: FinalRecipeOption, new_ingredient_name: str,
                   quantity: float, ingredient_data: Dict[str, IngredientInfo]) -> FinalRecipeOption:
    """
    Aggiunge un nuovo ingrediente alla ricetta.

    Args:
        recipe: Ricetta da modificare
        new_ingredient_name: Nome del nuovo ingrediente
        quantity: Quantità in grammi
        ingredient_data: Database ingredienti

    Returns:
        Ricetta modificata
    """
    modified_recipe = deepcopy(recipe)

    # Crea nuovo ingrediente
    new_ingredient = RecipeIngredient(
        name=new_ingredient_name, quantity_g=quantity)
    modified_recipe.ingredients.append(new_ingredient)

    # Ricalcola valori nutrizionali
    updated_ingredients = calculate_ingredient_cho_contribution(
        modified_recipe.ingredients, ingredient_data
    )
    modified_recipe.ingredients = updated_ingredients

    # Aggiorna totali
    modified_recipe.total_cho = sum(
        ing.cho_contribution for ing in updated_ingredients if ing.cho_contribution is not None)
    modified_recipe.total_calories = sum(
        ing.calories_contribution for ing in updated_ingredients if ing.calories_contribution is not None)
    modified_recipe.total_protein_g = sum(
        ing.protein_contribution_g for ing in updated_ingredients if ing.protein_contribution_g is not None)
    modified_recipe.total_fat_g = sum(
        ing.fat_contribution_g for ing in updated_ingredients if ing.fat_contribution_g is not None)
    modified_recipe.total_fiber_g = sum(
        ing.fiber_contribution_g for ing in updated_ingredients if ing.fiber_contribution_g is not None)

    # Aggiorna flag dietetici
    return compute_dietary_flags(modified_recipe, ingredient_data)


def suggest_cho_adjustment(recipe: FinalRecipeOption, target_cho: float,
                           ingredient_data: Dict[str, IngredientInfo]) -> Optional[Tuple[str, str, float]]:
    """
    Suggerisce un aggiustamento per avvicinare la ricetta al target CHO.

    Strategie:
    1. Se serve aumentare CHO: suggerisce di aggiungere un nuovo ingrediente ricco di CHO
       o di aumentare la quantità di un ingrediente esistente
    2. Se serve ridurre CHO: suggerisce di ridurre la quantità dell'ingrediente
       con il maggior contributo di CHO

    Args:
        recipe: Ricetta da analizzare
        target_cho: Target CHO in grammi
        ingredient_data: Database ingredienti con informazioni nutrizionali

    Returns:
        Tupla (tipo_aggiustamento, nome_ingrediente, quantità) dove:
        - tipo_aggiustamento: "add" o "modify"
        - nome_ingrediente: nome dell'ingrediente da aggiungere o modificare
        - quantità: nuova quantità o quantità da aggiungere
        Ritorna None se non è possibile suggerire un aggiustamento valido
    """
    if recipe.total_cho is None or target_cho is None:
        return None

    cho_difference = target_cho - recipe.total_cho

    # Se differenza minima, non serve aggiustamento
    if abs(cho_difference) < 5:
        return None

    # Determina se aumentare o ridurre CHO
    if cho_difference > 0:
        # Dobbiamo aumentare CHO
        # Filtra ingredienti DB ricchi di CHO
        high_cho_ingredients = [(name, info) for name, info in ingredient_data.items()
                                if info.cho_per_100g > 20 and info.is_vegan == recipe.is_vegan
                                and info.is_vegetarian == recipe.is_vegetarian
                                and info.is_gluten_free == recipe.is_gluten_free
                                and info.is_lactose_free == recipe.is_lactose_free]

        if high_cho_ingredients:
            # Seleziona casualmente un ingrediente
            random.seed(42)  # Per riproducibilità
            chosen_name, chosen_info = random.choice(high_cho_ingredients)

            # Calcola quantità necessaria per aggiungere CHO mancanti
            qty_needed = (cho_difference / chosen_info.cho_per_100g) * 100
            qty_needed = max(10, min(100, qty_needed))  # Limita tra 10g e 100g

            # Verifica se l'ingrediente è già presente
            for ing in recipe.ingredients:
                if ing.name == chosen_name:
                    return ("modify", chosen_name, ing.quantity_g + qty_needed)

            # Altrimenti, suggerisci di aggiungerlo
            return ("add", chosen_name, qty_needed)
    else:
        # Dobbiamo ridurre CHO
        # Trova l'ingrediente con più alto contributo CHO
        max_contributor = None
        max_contribution = 0

        for ing in recipe.ingredients:
            if ing.cho_contribution and ing.cho_contribution > max_contribution:
                max_contributor = ing
                max_contribution = ing.cho_contribution

        if max_contributor:
            # Calcola di quanto ridurre la quantità
            # Limitando la riduzione al 60% per evitare di ridurre troppo
            cho_to_remove = abs(cho_difference)
            if max_contributor.name in ingredient_data:
                cho_per_g = ingredient_data[max_contributor.name].cho_per_100g / 100
                if cho_per_g > 0:
                    qty_to_remove = min(
                        cho_to_remove / cho_per_g, max_contributor.quantity_g * 0.6)
                    return ("modify", max_contributor.name, max_contributor.quantity_g - qty_to_remove)

    return None

# --- FUNZIONE PRINCIPALE ---


def verifier_agent(state: GraphState) -> GraphState:
    """
    Node Function: Verifica, ottimizza e corregge le ricette generate.
    Versione potenziata con verifica di diversità e correzione flag dietetici.
    """
    print("\n--- ESECUZIONE NODO: Verifica e Ottimizzazione Ricette ---")

    # Recupera componenti necessari dallo stato
    recipes_from_generator = state.get('generated_recipes', [])
    preferences = state.get('user_preferences')
    ingredient_data = state.get('available_ingredients_data')
    normalized_to_original = state.get('normalized_to_original', {})
    original_to_normalized = state.get(
        'original_to_normalized', {})  # Aggiungi questo
    faiss_index = state.get('faiss_index')
    index_to_name_mapping = state.get('index_to_name_mapping')
    embedding_model = state.get('embedding_model')
    # Dovrebbe essere normalize_name da utils
    normalize_function = state.get('normalize_function')

    # Aggiunta di debug per ispezionare il database ingredienti
    print(
        f"DEBUG: Database ingredienti contiene {len(ingredient_data)} elementi")
    print(
        f"DEBUG: Mapping normalizzato contiene {len(normalized_to_original)} elementi")
    print(
        f"DEBUG: Mapping inverso contiene {len(original_to_normalized)} elementi")

    # Validazione input essenziali
    if not recipes_from_generator:
        print("Errore Verifier: Nessuna ricetta ricevuta dal generatore.")
        state['error_message'] = "Nessuna ricetta generata da verificare."
        state['final_verified_recipes'] = []
        return state

    if not all([preferences, ingredient_data, faiss_index, index_to_name_mapping, embedding_model, normalize_function]):
        print("Errore Verifier: Componenti essenziali mancanti nello stato (prefs, db, faiss, model, etc.).")
        state['error_message'] = "Errore interno: Dati o componenti mancanti per la verifica."
        state['final_verified_recipes'] = []
        return state

    target_cho = preferences.target_cho
    # Tolleranza % per considerare una ricetta "nel range" dopo l'ottimizzazione iniziale
    # +/- 30% (più larga per dare chance all'ottimizzazione)
    fixed_cho_tolerance = 6.0
    min_cho_initial = target_cho - fixed_cho_tolerance
    max_cho_initial = target_cho + fixed_cho_tolerance

    print(
        f"Verifica di {len(recipes_from_generator)} ricette generate. Target CHO: {target_cho:.1f}g")
    print(
        f"Range CHO post-ottimizzazione iniziale target: {min_cho_initial:.1f} - {max_cho_initial:.1f}g")

    # --- FASE 1: MATCHING, CALCOLO NUTRIENTI E VERIFICA DIETETICA PRELIMINARE ---
    processed_recipes_phase1 = []
    print("\nFase 1: Matching Ingredienti, Calcolo Nutrienti e Verifica Dietetica Preliminare")

    for recipe_gen in recipes_from_generator:
        # 1. Match ingredienti e calcolo iniziale nutrienti
        recipe_matched, match_success = match_recipe_ingredients(
            recipe_gen,
            ingredient_data,
            normalized_to_original,
            original_to_normalized,  # Aggiungi questo parametro
            faiss_index,
            index_to_name_mapping,
            embedding_model,
            normalize_function
        )

        if not match_success:
            print(
                f"Ricetta '{recipe_gen.name}' scartata (Fase 1): Matching fallito o CHO non calcolabile.")
            continue

        # 2. Calcola/Verifica flag dietetici basati sul DB
        recipe_flags_computed = compute_dietary_flags(
            recipe_matched, ingredient_data)

        # 3. Verifica preliminare rispetto alle preferenze utente
        if not verify_dietary_preferences(recipe_flags_computed, preferences):
            print(
                f"Ricetta '{recipe_flags_computed.name}' scartata (Fase 1): Non rispetta le preferenze dietetiche.")
            continue

        # Se passa tutti i controlli della fase 1, aggiungila alla lista
        processed_recipes_phase1.append(recipe_flags_computed)

    if not processed_recipes_phase1:
        print(
            "Errore Verifier: Nessuna ricetta ha superato la Fase 1 (matching/dietetica).")
        state['error_message'] = "Nessuna ricetta valida dopo il matching iniziale e la verifica dietetica."
        state['final_verified_recipes'] = []
        return state
    print(
        f"Ricette che hanno superato la Fase 1: {len(processed_recipes_phase1)}")

    # --- FASE 2: OTTIMIZZAZIONE CHO ---
    processed_recipes_phase2 = []
    print("\nFase 2: Ottimizzazione CHO")

    for recipe_p1 in processed_recipes_phase1:
        # Controlla se CHO è valido prima di ottimizzare
        if recipe_p1.total_cho is None:
            print(
                f"Ricetta '{recipe_p1.name}' scartata (Fase 2): CHO non calcolato, impossibile ottimizzare.")
            continue

        # Verifica se è già nel range target INIZIALE
        is_in_initial_range = (
            min_cho_initial <= recipe_p1.total_cho <= max_cho_initial)

        if is_in_initial_range:
            print(
                f"Ricetta '{recipe_p1.name}' già nel range CHO iniziale ({recipe_p1.total_cho:.1f}g).")
            # Mantiene la ricetta così com'è
            processed_recipes_phase2.append(recipe_p1)
            continue

        # Se non è nel range, tenta l'ottimizzazione
        print(
            f"Ricetta '{recipe_p1.name}' fuori range iniziale ({recipe_p1.total_cho:.1f}g). Tento ottimizzazione...")
        optimized_recipe = optimize_recipe_cho(
            deepcopy(recipe_p1), target_cho, ingredient_data)

        if optimized_recipe and optimized_recipe.total_cho is not None:
            is_optimized_in_range = (
                min_cho_initial <= optimized_recipe.total_cho <= max_cho_initial)
            improved = abs(optimized_recipe.total_cho -
                           target_cho) < abs(recipe_p1.total_cho - target_cho)
            if is_optimized_in_range:
                print(
                    f" -> Ottimizzazione riuscita! Nuovo CHO: {optimized_recipe.total_cho:.1f}g (Nel range iniziale)")
                processed_recipes_phase2.append(optimized_recipe)
            elif improved:
                print(
                    f" -> Ottimizzazione parziale. Nuovo CHO: {optimized_recipe.total_cho:.1f}g (Migliorato ma fuori range iniziale)")
                processed_recipes_phase2.append(optimized_recipe)
            else:
                print(
                    f" -> Ottimizzazione non migliorativa (Nuovo CHO: {optimized_recipe.total_cho:.1f}g). Scarto ricetta.")
        else:
            print(
                f" -> Ottimizzazione base fallita per '{recipe_p1.name}'. Tento aggiustamento ADD/MODIFY...")
            adjustment_suggestion = suggest_cho_adjustment(
                recipe_p1, target_cho, ingredient_data)
            adjusted_recipe = None
            if adjustment_suggestion:
                action, ingredient_name_db, quantity = adjustment_suggestion
                if action == "add":
                    adjusted_recipe = add_ingredient(
                        deepcopy(recipe_p1), ingredient_name_db, quantity, ingredient_data)
                elif action == "modify":
                    target_ing_to_modify = None
                    for ing in recipe_p1.ingredients:
                        name = ing.name if ing.name and "Info Mancanti" not in ing.name else ing.original_llm_name
                        if name == ingredient_name_db:
                            target_ing_to_modify = ing
                            break
                    if target_ing_to_modify and target_ing_to_modify.quantity_g is not None:
                        if ingredient_name_db in ingredient_data and ingredient_data[ingredient_name_db].cho_per_100g is not None and ingredient_data[ingredient_name_db].cho_per_100g > 0.1:
                            cho_diff_for_tune = (quantity - target_ing_to_modify.quantity_g) * (
                                ingredient_data[ingredient_name_db].cho_per_100g / 100.0)
                            adjusted_recipe = fine_tune_recipe(
                                deepcopy(recipe_p1), target_ing_to_modify, cho_diff_for_tune, ingredient_data)
                        else:
                            print(
                                f"Errore (suggest-modify): Info CHO mancanti per '{ingredient_name_db}'")
                    else:
                        print(
                            f"Errore (suggest-modify): Ingrediente '{ingredient_name_db}' non trovato o qtà nulla in ricetta.")

            if adjusted_recipe and adjusted_recipe.total_cho is not None:
                is_adjusted_in_range = (
                    min_cho_initial <= adjusted_recipe.total_cho <= max_cho_initial)
                improved_drastic = abs(
                    adjusted_recipe.total_cho - target_cho) < abs(recipe_p1.total_cho - target_cho)
                if is_adjusted_in_range:
                    print(
                        f" -> Aggiustamento ADD/MODIFY riuscito! Nuovo CHO: {adjusted_recipe.total_cho:.1f}g (Nel range iniziale)")
                    processed_recipes_phase2.append(adjusted_recipe)
                elif improved_drastic:
                    print(
                        f" -> Aggiustamento ADD/MODIFY parziale. Nuovo CHO: {adjusted_recipe.total_cho:.1f}g (Migliorato ma fuori range iniziale)")
                    processed_recipes_phase2.append(adjusted_recipe)
                else:
                    print(
                        f" -> Aggiustamento ADD/MODIFY non migliorativo. Scarto ricetta.")
            else:
                print(
                    f" -> Ottimizzazione/Aggiustamento falliti definitivamente per '{recipe_p1.name}'. Scarto ricetta.")

    if not processed_recipes_phase2:
        print(
            "Errore Verifier: Nessuna ricetta ha superato la Fase 2 (ottimizzazione CHO).")
        state['error_message'] = "Nessuna ricetta è risultata valida o ottimizzabile per il target CHO."
        state['final_verified_recipes'] = []
        return state
    print(
        f"Ricette che hanno superato la Fase 2: {len(processed_recipes_phase2)}")

    # --- FASE 3: VERIFICA FINALE (QUALITÀ, REALISMO, RANGE STRETTO) ---
    processed_recipes_phase3 = []  # Cambiato nome variabile per chiarezza
    print("\nFase 3: Verifica Finale (Qualità, Realismo, Range CHO Stretto)")

    # Tolleranza % finale più stretta
    fixed_cho_tolerance = 6.0  # +/- 15%
    min_cho_final = target_cho - fixed_cho_tolerance
    max_cho_final = target_cho + fixed_cho_tolerance
    print(
        f"Range CHO finale target: {min_cho_final:.1f} - {max_cho_final:.1f}g")

    # Soglia quantità massima e ingredienti da escludere
    max_ingredient_quantity_g = 250.0
    quantity_check_exclusions = {
        "brodo vegetale", "acqua", "latte", "vino bianco", "brodo di pollo", "brodo di pesce",
        "passata di pomodoro", "polpa di pomodoro"
    }
    print(
        f"Controllo quantità massima per ingrediente solido: < {max_ingredient_quantity_g}g")

    # Usa la variabile corretta (processed_recipes_phase2) nel loop
    for recipe_p2 in processed_recipes_phase2:
        # a) Controllo numero minimo ingredienti
        if not recipe_p2.ingredients or len(recipe_p2.ingredients) < 3:
            print(
                f"Ricetta '{recipe_p2.name}' scartata (Fase 3): Meno di 3 ingredienti.")
            continue
        # b) Controllo numero minimo istruzioni
        if not recipe_p2.instructions or len(recipe_p2.instructions) < 2:
            print(
                f"Ricetta '{recipe_p2.name}' scartata (Fase 3): Meno di 2 istruzioni.")
            continue

        # c) *** INIZIO BLOCCO CONTROLLO QUANTITA' MASSIMA ***
        quantity_ok = True
        for ing in recipe_p2.ingredients:
            # Nome per controllo esclusione e stampa
            check_name = ing.name if ing.name and "Info Mancanti" not in ing.name else ing.original_llm_name
            # Controlla solo se il nome esiste e non è tra le esclusioni
            if check_name and check_name not in quantity_check_exclusions:
                # Controlla la quantità solo se è un numero valido
                if ing.quantity_g is not None and ing.quantity_g > max_ingredient_quantity_g:
                    print(
                        f"Ricetta '{recipe_p2.name}' scartata (Fase 3): Ingrediente '{check_name}' supera quantità massima ({ing.quantity_g:.1f}g > {max_ingredient_quantity_g:.1f}g)")
                    quantity_ok = False
                    break  # Esci dal loop interno
        if not quantity_ok:
            # Salta al prossimo ciclo del loop esterno (prossima ricetta)
            continue
        # *** FINE BLOCCO CONTROLLO QUANTITA' MASSIMA ***

        # d) Controllo range CHO finale (stretto)
        if not (recipe_p2.total_cho and min_cho_final <= recipe_p2.total_cho <= max_cho_final):
            print(
                f"Ricetta '{recipe_p2.name}' scartata (Fase 3): CHO={recipe_p2.total_cho:.1f}g fuori dal range finale ({min_cho_final:.1f}-{max_cho_final:.1f}g)")
            continue

        # e) Ri-verifica preferenze dietetiche (sicurezza)
        if not verify_dietary_preferences(recipe_p2, preferences):
            print(
                f"Ricetta '{recipe_p2.name}' scartata (Fase 3): Fallita verifica dietetica finale.")
            continue

        # Se passa tutti i controlli della fase 3
        print(
            f"Ricetta '{recipe_p2.name}' verificata (Fase 3) (CHO: {recipe_p2.total_cho:.1f}g, Ingredienti: {len(recipe_p2.ingredients)})")
        # Aggiungi alla lista di quelle che passano la fase 3
        processed_recipes_phase3.append(recipe_p2)

    if not processed_recipes_phase3:
        print("Errore Verifier: Nessuna ricetta ha superato la Fase 3 (verifiche finali).")
        state['error_message'] = "Nessuna ricetta ha superato i controlli finali di qualità e range CHO."
        state['final_verified_recipes'] = []
        return state
    print(
        f"Ricette che hanno superato la Fase 3: {len(processed_recipes_phase3)}")

    # --- FASE 4: VERIFICA DIVERSITÀ ---
    processed_recipes_phase4 = []  # Cambiato nome variabile
    if len(processed_recipes_phase3) > 1:
        print("\nFase 4: Verifica Diversità tra Ricette")
        similarity_thr = 0.65
        # Usa la lista corretta (processed_recipes_phase3) come input
        processed_recipes_phase4 = ensure_recipe_diversity(
            processed_recipes_phase3, target_cho, similarity_threshold=similarity_thr)
        print(
            f"Ricette diverse selezionate: {len(processed_recipes_phase4)} su {len(processed_recipes_phase3)} (Soglia: {similarity_thr})")
    else:
        # Se c'è solo una ricetta, passa direttamente
        processed_recipes_phase4 = processed_recipes_phase3

    if not processed_recipes_phase4:
        print("Errore Verifier: Nessuna ricetta rimasta dopo il controllo di diversità.")
        state['error_message'] = "Nessuna ricetta selezionata dopo il filtro di diversità."
        state['final_verified_recipes'] = []
        return state

    # --- FASE 5: SELEZIONE FINALE E ORDINAMENTO ---
    print("\nFase 5: Selezione Finale e Ordinamento")
    # Ordina le ricette diverse (processed_recipes_phase4) per vicinanza al target CHO
    processed_recipes_phase4.sort(key=lambda r: abs(
        r.total_cho - target_cho) if r.total_cho is not None else float('inf'))

    # Limita al numero massimo desiderato di ricette finali
    max_final_recipes = 3  # Puoi cambiare questo valore
    final_selected_recipes = processed_recipes_phase4[:max_final_recipes]
    print(
        f"Selezionate le migliori {len(final_selected_recipes)} ricette finali.")

    # --- AGGIORNA STATO FINALE ---
    state['final_verified_recipes'] = final_selected_recipes

    # Imposta messaggio di errore/successo nello stato
    if not final_selected_recipes:
        state['error_message'] = "Processo completato ma nessuna ricetta finale selezionata."
    elif len(final_selected_recipes) < max_final_recipes:
        state['error_message'] = f"Trovate solo {len(final_selected_recipes)} ricette finali (invece delle {max_final_recipes} desiderate). Potresti provare a rilassare i vincoli."
    else:
        state.pop('error_message', None)  # Rimuovi errore se successo pieno

    print(
        f"\n--- Verifica completata: {len(final_selected_recipes)} ricette finali selezionate ---")
    return state
