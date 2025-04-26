"""
Agente di verifica e ottimizzazione delle ricette generate.

Questo modulo implementa l'agente verificatore potenziato, responsabile dell'analisi,
matching, ottimizzazione e verifica delle ricette generate. Questo agente è il
"cervello" del sistema in grado di correggere e migliorare le ricette per soddisfare
i requisiti nutrizionali e dietetici.
"""
from typing import List, Dict, Optional, Tuple, Union
from copy import deepcopy
from enum import Enum, auto
import random
from ingredient_synonyms import FALLBACK_MAPPING


from model_schema import GraphState, FinalRecipeOption, UserPreferences, RecipeIngredient, IngredientInfo, CalculatedIngredient
from utils import find_best_match_faiss, calculate_ingredient_cho_contribution

# -- Refactor ---

# --- CLASSI DI SUPPORTO PER OTTIMIZZAZIONE ---


class OptimizationStrategy(Enum):
    """Enum per identificare le diverse strategie di ottimizzazione disponibili."""
    SINGLE_INGREDIENT = auto()    # Modifica un solo ingrediente chiave
    PROPORTIONAL = auto()         # Modifica proporzionalmente tutti gli ingredienti con CHO
    CASCADE = auto()              # Approccio a cascata (primari, secondari, minori)
    HYBRID = auto()               # Combina più strategie in sequenza


class OptimizationResult:
    """Classe che rappresenta il risultato di un tentativo di ottimizzazione."""

    def __init__(self,
                 recipe: FinalRecipeOption,
                 success: bool,
                 cho_improvement: float,
                 strategy_used: OptimizationStrategy,
                 message: str = ""):
        self.recipe = recipe
        self.success = success
        self.cho_improvement = cho_improvement  # Miglioramento assoluto in g di CHO
        self.strategy_used = strategy_used
        self.message = message

    @property
    def is_better(self) -> bool:
        """Verifica se c'è stato un miglioramento nella vicinanza al target CHO."""
        return self.cho_improvement > 0

# --- FUNZIONI DI OTTIMIZZAZIONE CHO ---


def classify_ingredients_by_cho(recipe: FinalRecipeOption) -> Dict[str, List[CalculatedIngredient]]:
    """
    Classifica gli ingredienti per contributo CHO relativo alla ricetta.

    Categorizza in:
    - primary: >30% del totale CHO
    - secondary: 10-30% del totale CHO
    - minor: <10% del totale CHO
    - non_cho: nessun contributo CHO

    Args:
        recipe: Ricetta da analizzare

    Returns:
        Dizionario con ingredienti classificati per categoria
    """
    total_cho = recipe.total_cho if recipe.total_cho else 0
    classified = {'primary': [], 'secondary': [], 'minor': [], 'non_cho': []}

    if total_cho <= 0:
        return classified

    for ing in recipe.ingredients:
        if not hasattr(ing, 'cho_contribution') or ing.cho_contribution is None:
            classified['non_cho'].append(ing)
            continue

        cho_percent = (ing.cho_contribution / total_cho) * 100

        if cho_percent > 30:
            classified['primary'].append(ing)
        elif cho_percent >= 10:
            classified['secondary'].append(ing)
        elif cho_percent > 0:
            classified['minor'].append(ing)
        else:
            classified['non_cho'].append(ing)

    # Ordina per contributo decrescente in ogni categoria
    for category in ['primary', 'secondary', 'minor']:
        classified[category].sort(
            key=lambda x: x.cho_contribution if x.cho_contribution is not None else 0,
            reverse=True
        )

    return classified


def recalculate_nutrition(recipe: FinalRecipeOption,
                          ingredient_data: Dict[str, IngredientInfo]) -> FinalRecipeOption:
    """
    Ricalcola tutti i valori nutrizionali di una ricetta dopo modifiche agli ingredienti.

    Funzione helper centralizzata per aggiornare i totali nutrizionali.

    Args:
        recipe: Ricetta da aggiornare
        ingredient_data: Database ingredienti

    Returns:
        Ricetta con valori nutrizionali aggiornati
    """
    updated_recipe = deepcopy(recipe)

    # Ricalcola i contributi nutrizionali
    updated_ingredients = calculate_ingredient_cho_contribution(
        updated_recipe.ingredients, ingredient_data
    )
    updated_recipe.ingredients = updated_ingredients

    # Aggiorna i totali
    updated_recipe.total_cho = sum(
        ing.cho_contribution for ing in updated_ingredients
        if ing.cho_contribution is not None)

    updated_recipe.total_calories = sum(
        ing.calories_contribution for ing in updated_ingredients
        if ing.calories_contribution is not None)

    updated_recipe.total_protein_g = sum(
        ing.protein_contribution_g for ing in updated_ingredients
        if ing.protein_contribution_g is not None)

    updated_recipe.total_fat_g = sum(
        ing.fat_contribution_g for ing in updated_ingredients
        if ing.fat_contribution_g is not None)

    updated_recipe.total_fiber_g = sum(
        ing.fiber_contribution_g for ing in updated_ingredients
        if ing.fiber_contribution_g is not None)

    return updated_recipe


def adjust_ingredient_quantity(ingredient: CalculatedIngredient,
                               new_quantity: float,
                               min_quantity: float = 5.0,
                               max_quantity: float = 300.0) -> CalculatedIngredient:
    """
    Modifica la quantità di un ingrediente applicando limiti di sicurezza.

    Funzione helper per garantire che le quantità degli ingredienti rimangano
    entro limiti realistici dopo le modifiche.

    Args:
        ingredient: Ingrediente da modificare
        new_quantity: Nuova quantità desiderata in grammi
        min_quantity: Quantità minima accettabile (default: 5g)
        max_quantity: Quantità massima accettabile (default: 300g)

    Returns:
        Ingrediente con quantità modificata e vincolata ai limiti
    """
    updated_ingredient = deepcopy(ingredient)
    updated_ingredient.quantity_g = max(
        min_quantity, min(max_quantity, new_quantity))
    return updated_ingredient


def optimize_single_ingredient(recipe: FinalRecipeOption,
                               target_cho: float,
                               ingredient_data: Dict[str, IngredientInfo]) -> OptimizationResult:
    """
    Ottimizza la ricetta modificando un singolo ingrediente ricco di CHO.

    Strategia adatta per piccole differenze di CHO dove è preferibile 
    una modifica chirurgica a un solo ingrediente.

    Args:
        recipe: Ricetta da ottimizzare
        target_cho: Target CHO in grammi
        ingredient_data: Database ingredienti

    Returns:
        OptimizationResult con il risultato dell'ottimizzazione
    """
    original_recipe = deepcopy(recipe)
    optimized_recipe = deepcopy(recipe)

    # Calcola la differenza corrente dal target
    current_cho = recipe.total_cho if recipe.total_cho is not None else 0
    cho_difference = target_cho - current_cho

    # Se la differenza è troppo grande, questa strategia non è adatta
    if abs(cho_difference) > 15:
        return OptimizationResult(
            recipe=original_recipe,
            success=False,
            cho_improvement=0,
            strategy_used=OptimizationStrategy.SINGLE_INGREDIENT,
            message="Differenza CHO troppo grande per ottimizzazione singolo ingrediente"
        )

    # Classifica gli ingredienti
    classified = classify_ingredients_by_cho(recipe)

    # Cerca l'ingrediente migliore da modificare
    ingredient_to_adjust = None

    # Prima cerca nei primari, poi nei secondari se necessario
    for category in ['primary', 'secondary']:
        if classified[category] and not ingredient_to_adjust:
            for ing in classified[category]:
                if ing.name in ingredient_data and ingredient_data[ing.name].cho_per_100g > 0:
                    ingredient_to_adjust = ing
                    break

    # Se non troviamo un ingrediente adatto, fallisci
    if not ingredient_to_adjust or ingredient_to_adjust.name not in ingredient_data:
        return OptimizationResult(
            recipe=original_recipe,
            success=False,
            cho_improvement=0,
            strategy_used=OptimizationStrategy.SINGLE_INGREDIENT,
            message="Nessun ingrediente adatto trovato per l'ottimizzazione"
        )

    # Calcola la nuova quantità dell'ingrediente
    cho_per_100g = ingredient_data[ingredient_to_adjust.name].cho_per_100g
    if cho_per_100g <= 0:
        return OptimizationResult(
            recipe=original_recipe,
            success=False,
            cho_improvement=0,
            strategy_used=OptimizationStrategy.SINGLE_INGREDIENT,
            message=f"Ingrediente '{ingredient_to_adjust.name}' non ha CHO per adeguamento"
        )

    # Calcola la variazione di quantità necessaria
    weight_change = (cho_difference / cho_per_100g) * 100
    original_quantity = ingredient_to_adjust.quantity_g
    new_quantity = original_quantity + weight_change

    # Modifica l'ingrediente selezionato nella ricetta
    for i, ing in enumerate(optimized_recipe.ingredients):
        if ing.name == ingredient_to_adjust.name:
            optimized_recipe.ingredients[i] = adjust_ingredient_quantity(
                ing, new_quantity, min_quantity=5.0, max_quantity=300.0
            )
            break

    # Ricalcola i valori nutrizionali
    optimized_recipe = recalculate_nutrition(optimized_recipe, ingredient_data)

    # Verifica se c'è stato un miglioramento
    new_cho = optimized_recipe.total_cho if optimized_recipe.total_cho is not None else 0
    original_difference = abs(current_cho - target_cho)
    new_difference = abs(new_cho - target_cho)
    improvement = original_difference - new_difference

    return OptimizationResult(
        recipe=optimized_recipe,
        success=(improvement > 0),
        cho_improvement=improvement,
        strategy_used=OptimizationStrategy.SINGLE_INGREDIENT,
        message=f"Ingrediente '{ingredient_to_adjust.name}' modificato da {original_quantity:.1f}g a {optimized_recipe.ingredients[i].quantity_g:.1f}g"
    )


def optimize_proportionally(recipe: FinalRecipeOption,
                            target_cho: float,
                            ingredient_data: Dict[str, IngredientInfo]) -> OptimizationResult:
    """
    Ottimizza la ricetta applicando un fattore di scala a tutti gli ingredienti ricchi di CHO.

    Strategia adatta per differenze di CHO moderata-grande dove è preferibile
    mantenere le proporzioni della ricetta.

    Args:
        recipe: Ricetta da ottimizzare
        target_cho: Target CHO in grammi
        ingredient_data: Database ingredienti

    Returns:
        OptimizationResult con il risultato dell'ottimizzazione
    """
    original_recipe = deepcopy(recipe)
    optimized_recipe = deepcopy(recipe)

    # Calcola la differenza corrente dal target
    current_cho = recipe.total_cho if recipe.total_cho is not None else 0

    # Se non c'è CHO o è troppo basso, questa strategia non funzionerà bene
    if current_cho < 5:
        return OptimizationResult(
            recipe=original_recipe,
            success=False,
            cho_improvement=0,
            strategy_used=OptimizationStrategy.PROPORTIONAL,
            message="CHO totale troppo basso per scala proporzionale"
        )

    cho_difference = target_cho - current_cho

    # Calcola il fattore di scala
    # Limita il fattore per evitare modifiche troppo estreme
    if cho_difference > 0:  # Aumenta CHO
        ideal_scaling = 1 + (cho_difference / current_cho)
        scaling_factor = min(1.5, ideal_scaling)  # Limita a +50%
    else:  # Riduci CHO
        ideal_scaling = 1 + (cho_difference / current_cho)  # Sarà < 1
        scaling_factor = max(0.5, ideal_scaling)  # Limita a -50%

    # Classifica gli ingredienti
    classified = classify_ingredients_by_cho(recipe)

    # Combina primary e secondary per lo scaling
    cho_contributors = classified['primary'] + classified['secondary']
    if not cho_contributors:
        # Se non ci sono ingredienti significativi, prova anche con i minori
        cho_contributors = classified['minor']

    if not cho_contributors:
        return OptimizationResult(
            recipe=original_recipe,
            success=False,
            cho_improvement=0,
            strategy_used=OptimizationStrategy.PROPORTIONAL,
            message="Nessun ingrediente CHO significativo trovato per scaling"
        )

    changes_made = []

    # Applica il fattore di scala a tutti i contributori CHO
    for contributor in cho_contributors:
        for i, ing in enumerate(optimized_recipe.ingredients):
            if ing.name == contributor.name:
                original_qty = ing.quantity_g
                optimized_recipe.ingredients[i] = adjust_ingredient_quantity(
                    ing, original_qty * scaling_factor
                )
                changes_made.append(
                    f"'{ing.name}': {original_qty:.1f}g → {optimized_recipe.ingredients[i].quantity_g:.1f}g"
                )
                break

    # Ricalcola i valori nutrizionali
    optimized_recipe = recalculate_nutrition(optimized_recipe, ingredient_data)

    # Verifica se c'è stato un miglioramento
    new_cho = optimized_recipe.total_cho if optimized_recipe.total_cho is not None else 0
    original_difference = abs(current_cho - target_cho)
    new_difference = abs(new_cho - target_cho)
    improvement = original_difference - new_difference

    message = f"Scaling proporzionale (fattore {scaling_factor:.2f}) applicato a {len(changes_made)} ingredienti"

    return OptimizationResult(
        recipe=optimized_recipe,
        success=(improvement > 0),
        cho_improvement=improvement,
        strategy_used=OptimizationStrategy.PROPORTIONAL,
        message=message
    )


def optimize_cascade(recipe: FinalRecipeOption,
                     target_cho: float,
                     ingredient_data: Dict[str, IngredientInfo]) -> OptimizationResult:
    """
    Ottimizza la ricetta usando un approccio a cascata.

    Modifica prima gli ingredienti primari, poi i secondari, infine i minori se necessario,
    utilizzando fattori di scala diversi per ogni livello.

    Args:
        recipe: Ricetta da ottimizzare
        target_cho: Target CHO in grammi
        ingredient_data: Database ingredienti

    Returns:
        OptimizationResult con il risultato dell'ottimizzazione
    """
    original_recipe = deepcopy(recipe)
    optimized_recipe = deepcopy(recipe)

    # Calcola la differenza corrente dal target
    current_cho = recipe.total_cho if recipe.total_cho is not None else 0
    cho_difference = target_cho - current_cho

    # Classifica gli ingredienti
    classified = classify_ingredients_by_cho(optimized_recipe)
    changes_made = []

    # FASE 1: Modifica ingredienti primari
    if classified['primary']:
        # Calcola fattore di scala per ingredienti primari
        primary_cho = sum(ing.cho_contribution for ing in classified['primary']
                          if ing.cho_contribution is not None)

        if primary_cho > 0:
            primary_scaling = 1 + (cho_difference / primary_cho)
            # Limita il fattore per primari
            if cho_difference > 0:  # Aumenta
                primary_scaling = min(1.7, primary_scaling)
            else:  # Diminuisci
                primary_scaling = max(0.4, primary_scaling)

            # Applica scaling ai primari
            for ing in classified['primary']:
                for i, recipe_ing in enumerate(optimized_recipe.ingredients):
                    if recipe_ing.name == ing.name:
                        original_qty = recipe_ing.quantity_g
                        optimized_recipe.ingredients[i] = adjust_ingredient_quantity(
                            recipe_ing, original_qty * primary_scaling
                        )
                        changes_made.append(
                            f"Primario '{ing.name}': {original_qty:.1f}g → {optimized_recipe.ingredients[i].quantity_g:.1f}g"
                        )
                        break

            # Ricalcola dopo aver modificato i primari
            optimized_recipe = recalculate_nutrition(
                optimized_recipe, ingredient_data)
            # Aggiorna la differenza per le fasi successive
            current_cho = optimized_recipe.total_cho if optimized_recipe.total_cho is not None else 0
            cho_difference = target_cho - current_cho

    # FASE 2: Se necessario, modifica ingredienti secondari
    if abs(cho_difference) > 3 and classified['secondary']:
        # Ricalcola classificazione con i nuovi valori
        classified = classify_ingredients_by_cho(optimized_recipe)

        secondary_cho = sum(ing.cho_contribution for ing in classified['secondary']
                            if ing.cho_contribution is not None)

        if secondary_cho > 0:
            secondary_scaling = 1 + (cho_difference / secondary_cho)
            # Limita il fattore per secondari (meno estremo dei primari)
            if cho_difference > 0:
                secondary_scaling = min(1.4, secondary_scaling)
            else:
                secondary_scaling = max(0.6, secondary_scaling)

            # Applica scaling ai secondari
            for ing in classified['secondary']:
                for i, recipe_ing in enumerate(optimized_recipe.ingredients):
                    if recipe_ing.name == ing.name:
                        original_qty = recipe_ing.quantity_g
                        optimized_recipe.ingredients[i] = adjust_ingredient_quantity(
                            recipe_ing, original_qty * secondary_scaling
                        )
                        changes_made.append(
                            f"Secondario '{ing.name}': {original_qty:.1f}g → {optimized_recipe.ingredients[i].quantity_g:.1f}g"
                        )
                        break

            # Ricalcola dopo aver modificato i secondari
            optimized_recipe = recalculate_nutrition(
                optimized_recipe, ingredient_data)
            # Aggiorna la differenza per la fase successiva
            current_cho = optimized_recipe.total_cho if optimized_recipe.total_cho is not None else 0
            cho_difference = target_cho - current_cho

    # FASE 3: Se ancora necessario, modifica ingredienti minori
    if abs(cho_difference) > 5 and classified['minor']:
        # Ricalcola classificazione con i nuovi valori
        classified = classify_ingredients_by_cho(optimized_recipe)

        minor_cho = sum(ing.cho_contribution for ing in classified['minor']
                        if ing.cho_contribution is not None)

        if minor_cho > 0:
            minor_scaling = 1 + (cho_difference / minor_cho)
            # Limita il fattore per minori (cambiamenti modesti)
            if cho_difference > 0:
                minor_scaling = min(1.3, minor_scaling)
            else:
                minor_scaling = max(0.7, minor_scaling)

            # Applica scaling ai minori
            for ing in classified['minor']:
                for i, recipe_ing in enumerate(optimized_recipe.ingredients):
                    if recipe_ing.name == ing.name:
                        original_qty = recipe_ing.quantity_g
                        optimized_recipe.ingredients[i] = adjust_ingredient_quantity(
                            recipe_ing, original_qty * minor_scaling
                        )
                        changes_made.append(
                            f"Minore '{ing.name}': {original_qty:.1f}g → {optimized_recipe.ingredients[i].quantity_g:.1f}g"
                        )
                        break

            # Ricalcola finale
            optimized_recipe = recalculate_nutrition(
                optimized_recipe, ingredient_data)

    # Verifica se c'è stato un miglioramento
    new_cho = optimized_recipe.total_cho if optimized_recipe.total_cho is not None else 0
    original_difference = abs(
        recipe.total_cho - target_cho) if recipe.total_cho is not None else float('inf')
    new_difference = abs(new_cho - target_cho)
    improvement = original_difference - new_difference

    message = f"Ottimizzazione cascata con {len(changes_made)} modifiche: {original_recipe.total_cho:.1f}g → {new_cho:.1f}g"

    return OptimizationResult(
        recipe=optimized_recipe,
        success=(improvement > 0 and len(changes_made) > 0),
        cho_improvement=improvement,
        strategy_used=OptimizationStrategy.CASCADE,
        message=message
    )


def optimize_recipe_cho(recipe: FinalRecipeOption,
                        target_cho: float,
                        ingredient_data: Dict[str, IngredientInfo],
                        tolerance: float = 5.0) -> FinalRecipeOption:
    """
    Ottimizza una ricetta per avvicinarla al target CHO usando una strategia multi-approccio.

    Funzione principale unificata che sostituisce le diverse funzioni di ottimizzazione
    nel verifier_agent.py originale.

    Args:
        recipe: Ricetta da ottimizzare
        target_cho: Target CHO in grammi
        ingredient_data: Database ingredienti
        tolerance: Tolleranza accettabile in grammi di CHO (default: 5g)

    Returns:
        Ricetta ottimizzata o la ricetta originale se non sono possibili miglioramenti
    """
    # Controllo iniziale
    if recipe.total_cho is None:
        print(
            f"Impossibile ottimizzare: CHO totale non calcolato per '{recipe.name}'")
        return recipe

    # Se già nel range, non c'è bisogno di ottimizzazione
    if abs(recipe.total_cho - target_cho) <= tolerance:
        print(
            f"Ricetta '{recipe.name}' già nel range target (CHO: {recipe.total_cho:.1f}g, Target: {target_cho:.1f}g)")
        return recipe

    print(
        f"Ottimizzazione ricetta '{recipe.name}' - CHO attuale: {recipe.total_cho:.1f}g, Target: {target_cho:.1f}g")

    # Copia ricetta originale per confronto
    original_recipe = deepcopy(recipe)
    best_recipe = original_recipe
    best_improvement = 0

    # Calcola la differenza percentuale
    cho_difference = target_cho - recipe.total_cho
    difference_percentage = abs(cho_difference) / max(target_cho, 1) * 100

    # Strategia 1: Per piccole differenze, prova ottimizzazione di un singolo ingrediente
    if abs(cho_difference) < 15:
        print("Strategie 1: Ottimizzazione singolo ingrediente")
        result = optimize_single_ingredient(
            recipe, target_cho, ingredient_data)
        if result.success and result.cho_improvement > best_improvement:
            best_recipe = result.recipe
            best_improvement = result.cho_improvement
            print(f"Miglioramento con singolo ingrediente: {result.message}")

    # Strategia 2: Per differenze moderate, prova scala proporzionale
    if difference_percentage < 40:
        print("Strategia 2: Scaling proporzionale")
        result = optimize_proportionally(recipe, target_cho, ingredient_data)
        if result.success and result.cho_improvement > best_improvement:
            best_recipe = result.recipe
            best_improvement = result.cho_improvement
            print(f"Miglioramento con scaling proporzionale: {result.message}")

    # Strategia 3: Per grandi differenze, prova approccio a cascata
    if difference_percentage >= 25:
        print("Strategia 3: Ottimizzazione a cascata")
        result = optimize_cascade(recipe, target_cho, ingredient_data)
        if result.success and result.cho_improvement > best_improvement:
            best_recipe = result.recipe
            best_improvement = result.cho_improvement
            print(f"Miglioramento con cascata: {result.message}")

    # Verifica finale
    best_cho = best_recipe.total_cho if best_recipe.total_cho is not None else 0
    original_cho = original_recipe.total_cho if original_recipe.total_cho is not None else 0

    # Se il miglioramento è significativo, aggiorna il nome per indicare l'ottimizzazione
    if best_improvement > 0 and abs(original_cho - best_cho) > 10:
        best_recipe.name = f"{recipe.name} (Ottimizzata)"

    print(f"Risultato ottimizzazione CHO: {original_cho:.1f}g → {best_cho:.1f}g " +
          f"(Target: {target_cho:.1f}g, Miglioramento: {best_improvement:.1f}g)")

    return best_recipe

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
            if normalized_matched in FALLBACK_MAPPING:
                fallback_name = FALLBACK_MAPPING[normalized_matched]
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
            if normalized_original in FALLBACK_MAPPING:
                fallback_name = FALLBACK_MAPPING[normalized_original]
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
            if normalized_original in FALLBACK_MAPPING:
                fallback_name = FALLBACK_MAPPING[normalized_original]
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


def analyze_recipe_dietary_properties(
    recipe: FinalRecipeOption,
    ingredient_data: Dict[str, IngredientInfo] = None
) -> Tuple[bool, bool, bool, bool]:
    """
    Analizza una ricetta e determina le sue proprietà dietetiche (vegan, vegetarian, ecc.)
    basandosi sugli ingredienti e/o sui flag esistenti.

    La funzione supporta due modalità:
    1. Analisi basata sul database degli ingredienti (se ingredient_data è fornito)
    2. Analisi basata su liste di ingredienti non compatibili con varie diete
       (fallback quando ingredient_data non è disponibile o un ingrediente non è nel database)

    Args:
        recipe: Ricetta da analizzare
        ingredient_data: Opzionale, database degli ingredienti con informazioni nutrizionali

    Returns:
        Tupla con 4 booleani (is_vegan, is_vegetarian, is_gluten_free, is_lactose_free)
    """
    # Inizializza tutti i flag a True (cambieranno a False se troviamo ingredienti incompatibili)
    is_vegan = True
    is_vegetarian = True
    is_gluten_free = True
    is_lactose_free = True

    # Lista di ingredienti NON vegani
    non_vegan_ingredients = {
        "pollo", "tacchino", "manzo", "vitello", "maiale", "prosciutto",
        "pancetta", "salmone", "tonno", "pesce", "uova", "uovo", "formaggio",
        "parmigiano", "mozzarella", "ricotta", "burro", "latte", "panna"
    }

    # Lista di ingredienti NON vegetariani
    non_vegetarian_ingredients = {
        "pollo", "tacchino", "manzo", "vitello", "maiale", "prosciutto",
        "pancetta", "salmone", "tonno", "pesce"
    }

    # Lista di ingredienti NON senza glutine
    gluten_ingredients = {
        "pasta", "pane", "farina", "couscous", "orzo", "farro",
        "seitan", "pangrattato", "grano"
    }

    # Lista di ingredienti NON senza lattosio
    lactose_ingredients = {
        "latte", "formaggio", "parmigiano", "mozzarella", "ricotta",
        "burro", "panna", "yogurt"
    }

    # Se abbiamo i dati degli ingredienti, usiamo quelli
    if ingredient_data:
        # Verifica basata sui dati degli ingredienti dal database
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

    # In ogni caso, fai anche un controllo basato sui nomi (per maggiore sicurezza)
    # Questo è particolarmente utile per ingredienti che potrebbero non essere nel DB
    ing_names_lower = [ing.name.lower() for ing in recipe.ingredients]
    combined_text = " ".join(ing_names_lower).lower()

    # Controlli diretti sui nomi degli ingredienti
    for item in non_vegan_ingredients:
        if item in combined_text:
            is_vegan = False
            break

    for item in non_vegetarian_ingredients:
        if item in combined_text:
            is_vegetarian = False
            break

    for item in gluten_ingredients:
        if item in combined_text:
            is_gluten_free = False
            break

    for item in lactose_ingredients:
        if item in combined_text:
            is_lactose_free = False
            break

    # La regola principale: se un piatto non è vegano, non può essere vegetariano
    if not is_vegan and is_vegetarian:
        # Verifica più attentamente se è davvero vegetariano
        for ing in non_vegetarian_ingredients:
            if ing in combined_text:
                is_vegetarian = False
                break

    return is_vegan, is_vegetarian, is_gluten_free, is_lactose_free


def check_dietary_compatibility(recipe: FinalRecipeOption, preferences: UserPreferences) -> bool:
    """
    Verifica che la ricetta soddisfi le preferenze dietetiche dell'utente.

    Questa funzione è più concisa e mirata rispetto all'originale verify_dietary_preferences.

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


def update_recipe_dietary_flags(
    recipe: FinalRecipeOption,
    ingredient_data: Optional[Dict[str, IngredientInfo]] = None
) -> FinalRecipeOption:
    """
    Aggiorna i flag dietetici di una ricetta in base ai suoi ingredienti.

    Questa funzione sostituisce compute_dietary_flags e correct_dietary_flags,
    unificando la logica di aggiornamento dei flag.

    Args:
        recipe: Ricetta da aggiornare
        ingredient_data: Opzionale, database ingredienti con info nutrizionali

    Returns:
        Ricetta con flag dietetici aggiornati
    """
    updated_recipe = deepcopy(recipe)

    # Analizza le proprietà dietetiche della ricetta
    is_vegan, is_vegetarian, is_gluten_free, is_lactose_free = analyze_recipe_dietary_properties(
        recipe, ingredient_data
    )

    # Aggiorna i flag nella ricetta
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

    # Ricalcola valori nutrizionali utilizzando la nuova funzione centralizzata
    modified_recipe = recalculate_nutrition(modified_recipe, ingredient_data)

    # Aggiorna flag dietetici
    # Nota: abbiamo anche sostituito compute_dietary_flags con update_recipe_dietary_flags
    return update_recipe_dietary_flags(modified_recipe, ingredient_data)


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
        recipe_flags_computed = update_recipe_dietary_flags(
            recipe_matched, ingredient_data)

        # 3. Verifica preliminare rispetto alle preferenze utente
        if not check_dietary_compatibility(recipe_flags_computed, preferences):
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
        if not check_dietary_compatibility(recipe_p2, preferences):
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
