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
    Calcola un punteggio di somiglianza tra due ricette.

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
    Filtra una lista di ricette per assicurarsi che non ci siano ricette troppo simili.

    Args:
        recipes: Lista di ricette da filtrare
        similarity_threshold: Soglia sopra la quale le ricette sono considerate troppo simili

    Returns:
        Lista di ricette filtrata per diversità
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
    Corregge i flag dietetici di una ricetta basandosi sugli ingredienti.

    Args:
        recipe: Ricetta da verificare
        ingredient_data: Database ingredienti

    Returns:
        Ricetta con flag dietetici corretti
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
    Identifica gli ingredienti che contribuiscono maggiormente ai CHO.

    Args:
        recipe: Ricetta da analizzare
        ingredient_data: Database ingredienti

    Returns:
        Lista di ingredienti ordinati per contributo CHO (dal maggiore al minore)
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

    Args:
        recipe: Ricetta da aggiustare
        ingredient_to_adjust: Ingrediente da modificare
        cho_difference: Differenza di CHO da compensare
        ingredient_data: Database ingredienti

    Returns:
        Ricetta modificata
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
    Aggiusta proporzionalmente tutti gli ingredienti ricchi di CHO.

    Args:
        recipe: Ricetta da aggiustare
        cho_contributors: Lista di ingredienti ricchi di CHO
        scaling_factor: Fattore di scala da applicare
        ingredient_data: Database ingredienti

    Returns:
        Ricetta modificata
    """
    adjusted_recipe = deepcopy(recipe)
    contributor_names = [ing.name for ing in cho_contributors]

    # Applica scaling a tutti gli ingredienti CHO
    for i, ing in enumerate(adjusted_recipe.ingredients):
        if ing.name in contributor_names:
            original_quantity = ing.quantity_g
            # Applica scaling con limiti
            new_quantity = max(5, min(300, original_quantity * scaling_factor))
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


def optimize_recipe_cho(recipe: FinalRecipeOption, target_cho: float, ingredient_data: Dict[str, IngredientInfo]) -> Optional[FinalRecipeOption]:
    """
    Ottimizza una ricetta per raggiungere il target CHO.
    Utilizza diverse strategie in base alla situazione.

    Args:
        recipe: Ricetta da ottimizzare
        target_cho: Target CHO in grammi
        ingredient_data: Database ingredienti

    Returns:
        Ricetta ottimizzata o None se non ottimizzabile
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

    # 2. Identifica ingredienti ricchi di CHO
    cho_contributors = identify_cho_contributors(recipe, ingredient_data)
    if not cho_contributors:
        print(
            f"Ottimizzazione fallita: Nessun ingrediente ricco di CHO trovato in '{recipe.name}'")
        return None

    # 3. Scegli la strategia in base all'entità della differenza
    if abs(cho_difference) < 15:
        # Aggiustamento fine su ingrediente principale
        print(f"Strategia: Aggiustamento fine dell'ingrediente principale")
        return fine_tune_recipe(recipe, cho_contributors[0], cho_difference, ingredient_data)
    else:
        # Aggiustamento proporzionale di tutti gli ingredienti CHO
        print(f"Strategia: Scaling proporzionale di tutti gli ingredienti CHO")
        # Calcola fattore di scaling con limiti
        if current_cho > 0:  # Evita divisione per zero
            ideal_scaling = target_cho / current_cho
            # Limita il fattore di scaling per evitare cambiamenti troppo drastici
            if cho_difference > 0:  # Aumentare CHO
                scaling_factor = min(3.0, max(1.1, ideal_scaling))
            else:  # Ridurre CHO
                scaling_factor = max(0.4, min(0.9, ideal_scaling))

            return adjust_recipe_proportionally(recipe, cho_contributors, scaling_factor, ingredient_data)

    # Se arriviamo qui, non siamo riusciti a ottimizzare
    return None


def match_recipe_ingredients(recipe: FinalRecipeOption, ingredient_data: Dict[str, IngredientInfo],
                             faiss_index, index_to_name_mapping, embedding_model, normalize_function) -> Tuple[FinalRecipeOption, bool]:
    """
    Effettua il matching degli ingredienti della ricetta con il database usando FAISS.

    Args:
        recipe: Ricetta con ingredienti da matchare
        ingredient_data: Database ingredienti
        faiss_index: Indice FAISS
        index_to_name_mapping: Mapping indice-nome
        embedding_model: Modello di embedding
        normalize_function: Funzione di normalizzazione

    Returns:
        Tupla con (ricetta con ingredienti matchati, flag successo)
    """
    matched_recipe = deepcopy(recipe)
    matched_ingredients = []
    all_matched = True

    print(f"Matching ingredienti per ricetta '{recipe.name}'")

    for ing in recipe.ingredients:
        # Tenta il matching con FAISS
        match_result = find_best_match_faiss(
            llm_name=ing.name,
            faiss_index=faiss_index,
            index_to_name_mapping=index_to_name_mapping,
            model=embedding_model,
            normalize_func=normalize_function,
            threshold=0.60  # Soglia più bassa per aumentare le possibilità di match
        )

        if match_result:
            matched_db_name, match_score = match_result
            print(
                f"Ingrediente '{ing.name}' matchato a '{matched_db_name}' (score: {match_score:.2f})")

            # Crea nuovo ingrediente con nome matchato ma quantità originale
            matched_ingredients.append(
                RecipeIngredient(name=matched_db_name,
                                 quantity_g=ing.quantity_g)
            )
        else:
            print(f"Fallito matching per '{ing.name}'")
            # Keep the original ingredient but mark recipe as not fully matched
            matched_ingredients.append(ing)
            all_matched = False

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
    Può suggerire di aggiungere un nuovo ingrediente o modificare uno esistente.

    Args:
        recipe: Ricetta da analizzare
        target_cho: Target CHO in grammi
        ingredient_data: Database ingredienti

    Returns:
        Tupla (tipo_aggiustamento, nome_ingrediente, quantità) o None se non possibile
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
    print("--- ESECUZIONE NODO: Verifica e Ottimizzazione Ricette ---")

    # Recupera componenti necessari
    recipes = state.get('generated_recipes', [])
    preferences = state['user_preferences']
    ingredient_data = state.get('available_ingredients_data', {})
    faiss_index = state.get('faiss_index')
    index_to_name_mapping = state.get('index_to_name_mapping', [])
    embedding_model = state.get('embedding_model')
    normalize_function = state.get('normalize_function', normalize_name)

    if not recipes:
        state['error_message'] = "Nessuna ricetta da verificare."
        state['final_verified_recipes'] = []
        return state

    if not ingredient_data or not faiss_index or not index_to_name_mapping or not embedding_model:
        state['error_message'] = "Dati o componenti mancanti per la verifica."
        state['final_verified_recipes'] = []
        return state

    target_cho = preferences.target_cho
    cho_tolerance = 25.0  # Tolleranza più ampia per la prima verifica

    print(f"Verifica di {len(recipes)} ricette. Target CHO: {target_cho}g (Range: {target_cho-cho_tolerance} - {target_cho+cho_tolerance})")

    # --- FASE 1: MATCHING E CALCOLO NUTRIENTI ---
    matched_recipes = []
    print("Fase 1: Matching ingredienti e calcolo valori nutrizionali")

    for recipe in recipes:
        # Match degli ingredienti con il database
        matched_recipe, all_matched = match_recipe_ingredients(
            recipe, ingredient_data, faiss_index,
            index_to_name_mapping, embedding_model, normalize_function
        )

        if all_matched:
            # Calcola i flag dietetici in base agli ingredienti effettivi
            matched_recipe = compute_dietary_flags(
                matched_recipe, ingredient_data)
            # Aggiungi correzione aggiuntiva basata sulle liste di ingredienti
            matched_recipe = correct_dietary_flags(
                matched_recipe, ingredient_data)
            print(
                f"Ricetta '{matched_recipe.name}' completamente matchata: CHO={matched_recipe.total_cho:.1f}g")
            matched_recipes.append(matched_recipe)
        else:
            print(
                f"Ricetta '{recipe.name}' scartata: non tutti gli ingredienti sono stati matchati")

    if not matched_recipes:
        state['error_message'] = "Nessuna ricetta con ingredienti completamente matchati."
        state['final_verified_recipes'] = []
        return state

    # --- FASE 2: OTTIMIZZAZIONE CHO ---
    optimized_recipes = []
    print("Fase 2: Ottimizzazione CHO delle ricette")

    for recipe in matched_recipes:
        # Controlla se la ricetta è già nel range target
        is_in_range = (
            recipe.total_cho is not None and
            target_cho - cho_tolerance <= recipe.total_cho <= target_cho + cho_tolerance
        )

        if is_in_range:
            print(
                f"Ricetta '{recipe.name}' già nel range CHO: {recipe.total_cho:.1f}g")
            optimized_recipes.append(recipe)
            continue

        # Tenta l'ottimizzazione
        optimized = optimize_recipe_cho(recipe, target_cho, ingredient_data)
        if optimized and optimized.total_cho is not None:
            print(
                f"Ricetta '{recipe.name}' ottimizzata: CHO da {recipe.total_cho:.1f}g a {optimized.total_cho:.1f}g")
            optimized_recipes.append(optimized)
        else:
            # Tenta un approccio più drastico: aggiungi o rimuovi ingrediente
            adjustment = suggest_cho_adjustment(
                recipe, target_cho, ingredient_data)
            if adjustment:
                action, ingredient_name, new_quantity = adjustment
                if action == "add":
                    print(
                        f"Aggiungendo '{ingredient_name}' ({new_quantity:.1f}g) a '{recipe.name}'")
                    modified = add_ingredient(
                        recipe, ingredient_name, new_quantity, ingredient_data)
                    if modified and abs(modified.total_cho - target_cho) < abs(recipe.total_cho - target_cho):
                        print(
                            f"Ricetta '{recipe.name}' migliorata con aggiunta: CHO da {recipe.total_cho:.1f}g a {modified.total_cho:.1f}g")
                        optimized_recipes.append(modified)
                        continue

                # Se l'azione drastica non ha funzionato, manteniamo la ricetta originale
                print(
                    f"Impossibile ottimizzare '{recipe.name}', mantenuta originale")

            # Mantieni comunque la ricetta originale se non troppo lontana
            if recipe.total_cho and abs(recipe.total_cho - target_cho) < cho_tolerance * 1.5:
                print(
                    f"Ricetta '{recipe.name}' mantenuta nonostante CHO non ottimale: {recipe.total_cho:.1f}g")
                optimized_recipes.append(recipe)

    # --- FASE 3: VERIFICA FINALE E FILTRI ---
    verified_recipes = []
    print("Fase 3: Verifica finale e selezione ricette")

    # Verifica dietetica più stretta
    for recipe in optimized_recipes:
        # Verifica preferenze dietetiche
        if not verify_dietary_preferences(recipe, preferences):
            print(
                f"Ricetta '{recipe.name}' scartata: non rispetta preferenze dietetiche")
            continue

        # Verifica qualità generale
        if len(recipe.ingredients) < 3:
            print(
                f"Ricetta '{recipe.name}' scartata: troppo pochi ingredienti ({len(recipe.ingredients)})")
            continue

        # Verifica CHO finale più stringente
        final_cho_tolerance = 15.0  # Tolleranza più stretta per la selezione finale
        if not (recipe.total_cho and target_cho - final_cho_tolerance <= recipe.total_cho <= target_cho + final_cho_tolerance):
            print(
                f"Ricetta '{recipe.name}' scartata in fase finale: CHO={recipe.total_cho:.1f}g fuori dal range target")
            continue

        print(
            f"Ricetta '{recipe.name}' verificata (CHO: {recipe.total_cho:.1f}g, Ingredienti: {len(recipe.ingredients)})")
        verified_recipes.append(recipe)

    print(
        f"Ricette che hanno passato tutte le verifiche finali: {len(verified_recipes)}")

    # --- NUOVA FASE 4: VERIFICA DIVERSITÀ ---
    if len(verified_recipes) > 1:
        print("Fase 4: Verifica diversità delle ricette")
        diverse_recipes = ensure_recipe_diversity(
            verified_recipes, target_cho, similarity_threshold=0.6)
        print(
            f"Ricette diverse selezionate: {len(diverse_recipes)} su {len(verified_recipes)} valide")
    else:
        diverse_recipes = verified_recipes

    # --- FASE 5: SELEZIONE FINALE ---
    # Ordina per vicinanza al target CHO
    diverse_recipes.sort(key=lambda r: abs(
        r.total_cho - target_cho) if r.total_cho else float('inf'))

    # Limita a 5 ricette
    final_recipes = diverse_recipes[:5]

    # --- AGGIORNA STATO ---
    state['final_verified_recipes'] = final_recipes

    if not final_recipes:
        state['error_message'] = "Nessuna ricetta ha superato la verifica finale."
    elif len(final_recipes) < 3:
        state['error_message'] = f"Solo {len(final_recipes)} ricette hanno superato la verifica."
    else:
        # Rimuovi eventuali messaggi di errore precedenti
        state.pop('error_message', None)

    print(
        f"Verifica completata: {len(final_recipes)} ricette selezionate su {len(recipes)} iniziali")
    return state
