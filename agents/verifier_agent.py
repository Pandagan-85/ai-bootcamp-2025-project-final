"""
Agente di verifica delle ricette per il sistema di generazione ricette.

Questo modulo contiene funzioni per la verifica e l'ottimizzazione delle ricette generate.
Le funzioni principali includono:
1. Validazione degli ingredienti
2. Aggiustamento del contenuto di carboidrati (CHO)
3. Bilanciamento della distribuzione dei carboidrati tra gli ingredienti
4. Verifica finale delle ricette rispetto alle preferenze utente

Queste funzioni sono utilizzate per elaborare le ricette generate e garantire
che soddisfino al meglio i criteri dell'utente.
"""
from typing import List, Dict
from copy import deepcopy
from model_schema import GraphState, FinalRecipeOption, UserPreferences, RecipeIngredient

from utils import check_final_recipe_dietary_match, calculate_ingredient_cho_contribution


def validate_recipe_ingredients(recipe: FinalRecipeOption, ingredient_data: Dict) -> bool:
    """
    Verifica che tutti gli ingredienti nella ricetta esistano nel database degli ingredienti.

    Questa funzione controlla che ogni ingrediente nella ricetta sia presente nel dizionario
    ingredient_data, che contiene le informazioni nutrizionali degli ingredienti.

    Args:
        recipe: La ricetta da verificare
        ingredient_data: Dizionario con informazioni sugli ingredienti

    Returns:
        bool: True se tutti gli ingredienti sono validi, False altrimenti

    Note:
        - Un ingrediente è considerato non valido se il suo nome non è presente come chiave in ingredient_data
        - La funzione registra quali ingredienti sono risultati non validi per facilitare il debug
    """
    invalid_ingredients = []

    for ing in recipe.ingredients:
        if ing.name not in ingredient_data:
            invalid_ingredients.append(ing.name)

    if invalid_ingredients:
        print(
            f"Ricetta '{recipe.name}' contiene ingredienti non validi: {', '.join(invalid_ingredients)}")
        return False

    return True


def adjust_recipe_cho(recipe: FinalRecipeOption, target_cho: float, ingredient_data: Dict) -> FinalRecipeOption:
    """
    Aggiusta le quantità degli ingredienti per raggiungere il target CHO desiderato.

    Questa funzione analizza una ricetta e modifica le quantità degli ingredienti
    per avvicinare il suo contenuto totale di carboidrati al target specificato.
    L'aggiustamento viene fatto in modo intelligente, cercando di preservare le
    proporzioni della ricetta e bilanciare i cambiamenti.

    Implementa una versione "aggressiva" che può aggiustare anche ricette con
    CHO molto lontani dal target.

    Args:
        recipe: La ricetta da modificare
        target_cho: Il valore CHO target desiderato
        ingredient_data: Dizionario con informazioni nutrizionali degli ingredienti

    Returns:
        FinalRecipeOption: La ricetta modificata con quantità aggiustate o la ricetta
        originale se non sono necessarie modifiche o non è possibile aggiustarla

    Algoritmo:
    1. Se la ricetta è già vicina al target (±3g), la restituisce inalterata
    2. Identifica gli ingredienti ricchi di CHO che possono essere modificati
    3. Applica un fattore di scala globale o mirato in base alla differenza
    4. Aggiusta con precisione l'ingrediente principale se necessario
    5. Aggiorna il nome della ricetta per indicare che è stata modificata
    """
    # Se siamo già nel target, non fare nulla
    if abs(recipe.total_cho - target_cho) < 3.0:
        return recipe

    # Crea una copia profonda della ricetta per non modificare l'originale
    adjusted_recipe = deepcopy(recipe)

    # Identifica gli ingredienti ricchi di CHO (più del 5% del totale)
    cho_rich_ingredients = []
    for ing in adjusted_recipe.ingredients:
        if ing.cho_contribution > 0:
            # Se la ricetta ha pochi CHO totali, consideriamo tutti gli ingredienti con CHO
            if adjusted_recipe.total_cho < 30 or (ing.cho_contribution / adjusted_recipe.total_cho) > 0.05:
                cho_rich_ingredients.append(ing)

    # Se non ci sono ingredienti ricchi di CHO, verifica se ci sono nel database
    if not cho_rich_ingredients:
        # Verifica se ci sono altri ingredienti con CHO nel database che possiamo aumentare
        potential_ingredients = []
        for ing in adjusted_recipe.ingredients:
            if ing.name in ingredient_data and ingredient_data[ing.name].cho_per_100g > 5:
                potential_ingredients.append(ing)

        # Se troviamo ingredienti potenziali, usiamo quelli
        if potential_ingredients:
            cho_rich_ingredients = potential_ingredients

    # Se ancora non ci sono ingredienti ricchi di CHO, non possiamo modificare
    if not cho_rich_ingredients:
        return recipe

    # Ordina per contributo CHO decrescente
    cho_rich_ingredients.sort(key=lambda x: x.cho_contribution, reverse=True)

    # Calcola di quanto dobbiamo modificare i CHO
    cho_diff = target_cho - adjusted_recipe.total_cho

    # Calcola il fattore di scala necessario per tutta la ricetta
    if adjusted_recipe.total_cho > 0:
        # Se la ricetta ha un valore CHO significativo, possiamo semplicamente scalare tutto
        scaling_factor = target_cho / adjusted_recipe.total_cho

        # Limiti più permissivi per le ricette lontane dal target
        if abs(cho_diff) > 20:
            # Per casi estremi, consentiamo modifiche più drastiche
            min_scale = 0.5 if adjusted_recipe.total_cho > target_cho else 1.0
            max_scale = 1.0 if adjusted_recipe.total_cho > target_cho else 3.0
        else:
            # Per casi normali, limiti standard
            min_scale = 0.6
            max_scale = 2.0

        # Limita il fattore di scala entro i limiti
        scaling_factor = max(min_scale, min(scaling_factor, max_scale))

        # Applica il fattore di scala a tutti gli ingredienti ricchi di CHO
        recipe_ingredients = []
        for ing in adjusted_recipe.ingredients:
            # Applica scaling solo agli ingredienti ricchi di CHO
            if ing in cho_rich_ingredients:
                recipe_ingredients.append(
                    RecipeIngredient(
                        name=ing.name,
                        quantity_g=ing.quantity_g * scaling_factor
                    )
                )
            else:
                recipe_ingredients.append(
                    RecipeIngredient(
                        name=ing.name,
                        quantity_g=ing.quantity_g
                    )
                )

        # Ricalcola i contributi nutrizionali
        updated_ingredients = calculate_ingredient_cho_contribution(
            recipe_ingredients, ingredient_data)
        adjusted_recipe.ingredients = updated_ingredients
        adjusted_recipe.total_cho = round(
            sum(ing.cho_contribution for ing in updated_ingredients), 2)

    # Se necessario, fai un aggiustamento fine sull'ingrediente principale
    if abs(adjusted_recipe.total_cho - target_cho) > 5.0 and cho_rich_ingredients:
        main_ingredient = cho_rich_ingredients[0]
        ing_name = main_ingredient.name

        if ing_name in ingredient_data and ingredient_data[ing_name].cho_per_100g > 0:
            cho_per_100g = ingredient_data[ing_name].cho_per_100g

            # Ricalcola la differenza dopo il primo aggiustamento
            cho_diff = target_cho - adjusted_recipe.total_cho

            # Calcola la modifica necessaria
            gram_diff = (cho_diff / cho_per_100g) * 100.0

            # Trova l'ingrediente da modificare
            for i, ing in enumerate(adjusted_recipe.ingredients):
                if ing.name == ing_name:
                    # Calcola la nuova quantità
                    new_quantity = ing.quantity_g + gram_diff

                    # Assicurati che rimanga ragionevole (minimo 10g)
                    new_quantity = max(10.0, new_quantity)

                    # Aggiorna l'ingrediente
                    recipe_ingredients = [
                        RecipeIngredient(
                            name=ing.name,
                            quantity_g=new_quantity if ing.name == ing_name else ing.quantity_g
                        ) for ing in adjusted_recipe.ingredients
                    ]

                    # Ricalcola i contributi
                    updated_ingredients = calculate_ingredient_cho_contribution(
                        recipe_ingredients, ingredient_data)
                    adjusted_recipe.ingredients = updated_ingredients
                    adjusted_recipe.total_cho = round(
                        sum(ing.cho_contribution for ing in updated_ingredients), 2)
                    break

    # Aggiungi un'indicazione al nome che è stata modificata
    if recipe.total_cho != adjusted_recipe.total_cho:
        if "Aggiustata" not in recipe.name and "Bilanciata" not in recipe.name:
            adjusted_recipe.name = f"{adjusted_recipe.name} (Aggiustata)"

        # Se hai una descrizione, aggiorna anche quella
        if adjusted_recipe.description and "aggiustate" not in adjusted_recipe.description.lower():
            adjusted_recipe.description = f"{adjusted_recipe.description} Ricetta con quantità aggiustate per raggiungere il target CHO."

    return adjusted_recipe


def balance_cho_distribution(recipe: FinalRecipeOption, ingredient_data: Dict) -> FinalRecipeOption:
    """
    Bilancia la distribuzione dei CHO tra gli ingredienti di una ricetta.

    Questa funzione analizza se la ricetta ha un ingrediente dominante che fornisce una
    percentuale troppo alta del totale dei carboidrati. Se trovato, redistribuisce i carboidrati
    tra più ingredienti per creare una ricetta più bilanciata.

    Args:
        recipe: La ricetta da bilanciare
        ingredient_data: Dizionario con informazioni nutrizionali degli ingredienti

    Returns:
        FinalRecipeOption: La ricetta modificata con distribuzione di CHO più bilanciata
        o la ricetta originale se non è necessario bilanciarla

    Algoritmo:
    1. Identifica se esiste un ingrediente dominante (>90% dei CHO totali)
    2. Se non ci sono altri ingredienti con CHO, cerca potenziali ingredienti da aggiungere
    3. Riduce la quantità dell'ingrediente dominante e aumenta quella degli altri ingredienti
    4. Verifica se il bilanciamento ha migliorato la distribuzione
    """
    # Soglia massima accettabile per un singolo ingrediente
    MAX_CHO_PERCENTAGE = 0.9  # 90%

    # Verifica se c'è un ingrediente dominante
    dominant_ingredient = None
    for ing in recipe.ingredients:
        if recipe.total_cho > 0 and (ing.cho_contribution / recipe.total_cho) > MAX_CHO_PERCENTAGE:
            dominant_ingredient = ing
            break

    # Se non c'è un ingrediente dominante, non serve bilanciare
    if not dominant_ingredient:
        return recipe

    # Crea una copia della ricetta
    balanced_recipe = deepcopy(recipe)

    # Identifica altri ingredienti con carboidrati o potenziali da aggiungere
    other_cho_ingredients = []
    for ing in balanced_recipe.ingredients:
        if ing != dominant_ingredient and ing.name in ingredient_data:
            info = ingredient_data[ing.name]
            if info.cho_per_100g > 0:
                other_cho_ingredients.append(ing)

    # Se non ci sono altri ingredienti con CHO, cerca tra tutti quelli disponibili
    if not other_cho_ingredients:
        potential_ingredients = []
        for name, info in ingredient_data.items():
            # Verifica che non sia già presente e che abbia CHO
            if info.cho_per_100g > 0 and not any(ing.name == name for ing in balanced_recipe.ingredients):
                potential_ingredients.append({
                    "name": name,
                    "cho_per_100g": info.cho_per_100g
                })

        # Ordina per contenuto di CHO decrescente
        potential_ingredients.sort(
            key=lambda x: x["cho_per_100g"], reverse=True)

        # Prendiamo i primi 3 ingredienti con più CHO
        for i in range(min(3, len(potential_ingredients))):
            # Aggiungi un nuovo ingrediente con una quantità base di 30g
            new_ing = RecipeIngredient(
                name=potential_ingredients[i]["name"],
                quantity_g=30.0
            )
            recipe_ingredients = [
                RecipeIngredient(
                    name=ing.name,
                    quantity_g=ing.quantity_g
                ) for ing in balanced_recipe.ingredients
            ]
            recipe_ingredients.append(new_ing)

            # Ricalcola i contributi nutrizionali
            updated_ingredients = calculate_ingredient_cho_contribution(
                recipe_ingredients, ingredient_data)

            # Aggiorna gli ingredienti della ricetta
            balanced_recipe.ingredients = updated_ingredients

            # Ricalcola il CHO totale
            balanced_recipe.total_cho = round(
                sum(ing.cho_contribution for ing in updated_ingredients), 2)

            # Aggiungi all'elenco degli altri ingredienti CHO
            for ing in balanced_recipe.ingredients:
                if ing.name == potential_ingredients[i]["name"]:
                    other_cho_ingredients.append(ing)
                    break

    # Se ancora non abbiamo altri ingredienti CHO, dobbiamo accettare la ricetta com'è
    if not other_cho_ingredients:
        return recipe

    # Ottieni l'ingrediente dominante aggiornato (potrebbe essere cambiato dopo le aggiunte)
    for ing in balanced_recipe.ingredients:
        if ing.name == dominant_ingredient.name:
            dominant_ingredient = ing
            break

    # Calcola quanto CHO dobbiamo spostare dal dominante agli altri
    current_percentage = dominant_ingredient.cho_contribution / balanced_recipe.total_cho
    target_percentage = MAX_CHO_PERCENTAGE
    cho_to_redistribute = dominant_ingredient.cho_contribution - \
        (balanced_recipe.total_cho * target_percentage)

    # Se non c'è niente da redistribuire, restituisci la ricetta (potrebbe essere cambiata)
    if cho_to_redistribute <= 0:
        return balanced_recipe

    # Riduci la quantità dell'ingrediente dominante
    if dominant_ingredient.name in ingredient_data:
        cho_per_100g = ingredient_data[dominant_ingredient.name].cho_per_100g
        if cho_per_100g > 0:
            # Quanto dobbiamo ridurre in grammi?
            grams_to_reduce = (cho_to_redistribute / cho_per_100g) * 100.0

            # Assicurati di non ridurre troppo (max 40%)
            max_reduction = dominant_ingredient.quantity_g * 0.4
            grams_to_reduce = min(grams_to_reduce, max_reduction)

            # Applica la riduzione all'ingrediente dominante
            new_dominant_quantity = dominant_ingredient.quantity_g - grams_to_reduce

            # Aggiorna l'ingrediente dominante
            recipe_ingredients = [
                RecipeIngredient(
                    name=ing.name,
                    quantity_g=new_dominant_quantity if ing.name == dominant_ingredient.name else ing.quantity_g
                ) for ing in balanced_recipe.ingredients
            ]

            # Ricalcola i contributi
            updated_ingredients = calculate_ingredient_cho_contribution(
                recipe_ingredients, ingredient_data)
            balanced_recipe.ingredients = updated_ingredients
            balanced_recipe.total_cho = round(
                sum(ing.cho_contribution for ing in updated_ingredients), 2)

            # Aggiorna il riferimento all'ingrediente dominante
            for ing in balanced_recipe.ingredients:
                if ing.name == dominant_ingredient.name:
                    dominant_ingredient = ing
                    break

            # Distribuisci il CHO agli altri ingredienti
            cho_redistributed = 0.0

            # Calcola quanto CHO è stato rimosso dalla riduzione
            cho_removed = cho_to_redistribute - \
                (dominant_ingredient.cho_contribution -
                 (balanced_recipe.total_cho * target_percentage))

            # Distribuisci equamente il CHO rimosso tra gli altri ingredienti
            if other_cho_ingredients and cho_removed > 0:
                cho_per_ingredient = cho_removed / len(other_cho_ingredients)

                for target_ing in other_cho_ingredients:
                    if target_ing.name in ingredient_data:
                        target_cho_per_100g = ingredient_data[target_ing.name].cho_per_100g
                        if target_cho_per_100g > 0:
                            # Quanto dobbiamo aumentare in grammi?
                            grams_to_add = (
                                cho_per_ingredient / target_cho_per_100g) * 100.0

                            # Limite di aumento (max 100%)
                            max_increase = target_ing.quantity_g * 1.0
                            grams_to_add = min(grams_to_add, max_increase)

                            # Nuova quantità
                            new_quantity = target_ing.quantity_g + grams_to_add

                            # Aggiorna l'ingrediente
                            for i, ing in enumerate(balanced_recipe.ingredients):
                                if ing.name == target_ing.name:
                                    recipe_ingredient = RecipeIngredient(
                                        name=ing.name,
                                        quantity_g=new_quantity
                                    )
                                    break

                # Ricalcola tutto
                recipe_ingredients = [
                    RecipeIngredient(
                        name=ing.name,
                        quantity_g=ing.quantity_g
                    ) for ing in balanced_recipe.ingredients
                ]

                updated_ingredients = calculate_ingredient_cho_contribution(
                    recipe_ingredients, ingredient_data)
                balanced_recipe.ingredients = updated_ingredients
                balanced_recipe.total_cho = round(
                    sum(ing.cho_contribution for ing in updated_ingredients), 2)

    # Verifica se abbiamo effettivamente migliorato il bilanciamento
    max_percentage = 0.0
    for ing in balanced_recipe.ingredients:
        if balanced_recipe.total_cho > 0:
            percentage = ing.cho_contribution / balanced_recipe.total_cho
            max_percentage = max(max_percentage, percentage)

    # Aggiorna il nome e la descrizione per indicare il bilanciamento
    if recipe.name != balanced_recipe.name or max_percentage < current_percentage:
        if "Bilanciata" not in balanced_recipe.name:
            balanced_recipe.name = f"{balanced_recipe.name} (Bilanciata)"

        if balanced_recipe.description and "bilanciata" not in balanced_recipe.description.lower():
            balanced_recipe.description = f"{balanced_recipe.description} Ricetta con distribuzione di carboidrati bilanciata."

    return balanced_recipe


def verifier_agent(state: GraphState) -> GraphState:
    """
    Node Function: Verifica le ricette generate rispetto ai criteri finali.
    Implementato con logica Python, non usa LLM.
    """
    print("--- ESECUZIONE NODO: Verifica Finale Ricette ---")
    preferences: UserPreferences = state['user_preferences']
    generated_recipes: List[FinalRecipeOption] = state.get(
        'generated_recipes', [])
    ingredient_data = state['available_ingredients']
    target_cho = preferences.target_cho
    cho_tolerance = 10.0  # Tolleranza CHO
    exact_recipes_required = 3  # Numero esatto di ricette da fornire

    # Aumentiamo la soglia massima per un singolo ingrediente dal 75% al 90%
    max_cho_dominance = 0.9  # 90% invece di 75%

    if not generated_recipes:
        print("Nessuna ricetta generata da verificare.")
        state['final_verified_recipes'] = []
        # Potresti voler impostare un messaggio d'errore specifico qui
        if not state.get('error_message'):
            state['error_message'] = "Nessuna ricetta è stata generata con successo."
        return state

    valid_recipes: List[FinalRecipeOption] = []
    min_cho = target_cho - cho_tolerance
    max_cho = target_cho + cho_tolerance

    print(
        f"Verifica di {len(generated_recipes)} ricette generate. Target CHO: {target_cho}g (Range: {min_cho}g - {max_cho}g)")

    # NUOVO: Prima di tutto, verifica e scarta le ricette con ingredienti non validi
    filtered_recipes = []
    for recipe in generated_recipes:
        if validate_recipe_ingredients(recipe, ingredient_data):
            filtered_recipes.append(recipe)
        else:
            print(
                f"Ricetta '{recipe.name}' SCARTATA: Contiene ingredienti non presenti nel database.")

    print(
        f"Filtrate {len(filtered_recipes)}/{len(generated_recipes)} ricette con ingredienti validi.")

    # NUOVO: Salva le ricette originali per riferimento
    original_recipes = {}
    for recipe in filtered_recipes:
        original_recipes[recipe.name] = recipe

    # Fase 1: Prova a bilanciare le ricette che hanno un ingrediente troppo dominante
    balanced_recipes = []
    for recipe in filtered_recipes:
        # Salva il nome originale prima del bilanciamento
        original_name = recipe.name

        # Identifica ricette con ingredienti dominanti (>90% dei CHO)
        has_dominant_ingredient = False
        if recipe.total_cho > 0:
            for ing in recipe.ingredients:
                if (ing.cho_contribution / recipe.total_cho) > max_cho_dominance:
                    has_dominant_ingredient = True
                    break

        if has_dominant_ingredient:
            # Prova a bilanciare la distribuzione dei CHO
            balanced_recipe = balance_cho_distribution(recipe, ingredient_data)
            print(
                f"Ricetta '{recipe.name}' BILANCIATA per redistribuire i CHO più equamente.")

            # Aggiungi anche la versione bilanciata al dizionario originale
            original_recipes[balanced_recipe.name] = recipe

            balanced_recipes.append(balanced_recipe)
        else:
            balanced_recipes.append(recipe)

    # Fase 2: Prova ad aggiustare le ricette per avvicinarsi al target CHO
    adjusted_recipes = []
    for recipe in balanced_recipes:
        # Trova il nome base della ricetta (rimuovendo eventuali suffissi come "(Bilanciata)")
        base_name = recipe.name.split(" (")[0]

        # Cerca la ricetta originale
        original_recipe = None
        for orig_name, orig_recipe in original_recipes.items():
            if orig_name.split(" (")[0] == base_name:
                original_recipe = orig_recipe
                break

        # Se non troviamo l'originale, usa questa come riferimento
        if not original_recipe:
            original_recipe = recipe

        # MODIFICATO: Limita l'aggiustamento CHO a +/-50% del valore originale
        original_cho = original_recipe.total_cho
        # Non meno del 50% dell'originale
        min_allowed_cho = max(min_cho, original_cho * 0.5)
        # Non più del 150% dell'originale
        max_allowed_cho = min(max_cho, original_cho * 1.5)

        # Se è già nel range personalizzato, non modificare
        if min_allowed_cho <= recipe.total_cho <= max_allowed_cho:
            adjusted_recipes.append(recipe)
            continue

        # Tenta di aggiustare OGNI ricetta, anche quelle molto lontane
        adjusted_recipe = adjust_recipe_cho(
            recipe, target_cho, ingredient_data)
        print(
            f"Ricetta '{recipe.name}' AGGIUSTATA: CHO passato da {recipe.total_cho}g a {adjusted_recipe.total_cho}g (target: {target_cho}g)")
        adjusted_recipes.append(adjusted_recipe)

    # Fase 3: Verifica finale per tutte le ricette modificate
    for recipe in adjusted_recipes:
        # 1. Verifica CHO
        cho_ok = min_cho <= recipe.total_cho <= max_cho
        if not cho_ok:
            print(
                f"Ricetta '{recipe.name}' SCARTATA: CHO ({recipe.total_cho}g) fuori range ({min_cho}-{max_cho}g).")
            continue

        # 2. Verifica Dieta - MODIFICATO: considera solo le preferenze specificate dall'utente
        diet_ok = True
        if preferences.vegan and not recipe.is_vegan:
            diet_ok = False
        if preferences.vegetarian and not recipe.is_vegetarian:
            diet_ok = False
        if preferences.gluten_free and not recipe.is_gluten_free:
            diet_ok = False
        if preferences.lactose_free and not recipe.is_lactose_free:
            diet_ok = False

        if not diet_ok:
            # Costruisci un messaggio di log più specifico
            prefs_str = []
            if preferences.vegan:
                prefs_str.append("Vegano")
            if preferences.vegetarian and not preferences.vegan:
                prefs_str.append("Vegetariano")  # Evita ridondanza
            if preferences.gluten_free:
                prefs_str.append("Senza Glutine")
            if preferences.lactose_free:
                prefs_str.append("Senza Lattosio")
            print(f"Ricetta '{recipe.name}' SCARTATA: Non soddisfa le preferenze dietetiche richieste ({', '.join(prefs_str)}). Flag ricetta: V={recipe.is_vegan}, VG={recipe.is_vegetarian}, GF={recipe.is_gluten_free}, LF={recipe.is_lactose_free}")
            continue

        # 3. Verifica Equilibrio (Versione meno restrittiva)
        # Scarta ricette con meno di 3 ingredienti (troppo semplici)
        if len(recipe.ingredients) < 3:
            print(
                f"Ricetta '{recipe.name}' SCARTATA: Troppo semplice (solo {len(recipe.ingredients)} ingredienti).")
            continue

        # MODIFICATO: Non scartare ricette in base a ingredienti con CHO pari a 0
        # Solo verifica se sono troppo sbilanciati
        is_unbalanced = False
        if recipe.total_cho > 0:  # Evita divisione per zero
            for ing in recipe.ingredients:
                if (ing.cho_contribution / recipe.total_cho) > max_cho_dominance and len(recipe.ingredients) > 2:
                    print(
                        f"Ricetta '{recipe.name}' SCARTATA: Potenzialmente sbilanciata (un ingrediente fornisce >{int(max_cho_dominance*100)}% CHO).")
                    is_unbalanced = True
                    break  # Basta un ingrediente dominante

        if is_unbalanced:
            continue

        # Se tutti i check sono OK, aggiungi alla lista di ricette valide
        print(
            f"Ricetta '{recipe.name}' VERIFICATA (CHO: {recipe.total_cho}g, Dieta OK, Ingredienti: {len(recipe.ingredients)}).")
        valid_recipes.append(recipe)

    print(
        f"Ricette che hanno passato tutte le verifiche: {len(valid_recipes)}")

    # NUOVO: Assicurati che le ricette selezionate siano diverse
    verified_recipes = []
    recipe_types = set()  # Per tenere traccia dei tipi di ricette
    main_ingredients = set()  # Per tenere traccia degli ingredienti principali

    if valid_recipes:
        # Ordina le ricette in base alla vicinanza al target CHO
        valid_recipes.sort(key=lambda r: abs(r.total_cho - target_cho))

        # In caso di parità, favorisci ricette con più varietà di ingredienti
        for i in range(len(valid_recipes)-1):
            if abs(valid_recipes[i].total_cho - target_cho) == abs(valid_recipes[i+1].total_cho - target_cho):
                # Se due ricette hanno la stessa vicinanza al target CHO, ordina per numero di ingredienti
                if len(valid_recipes[i].ingredients) < len(valid_recipes[i+1].ingredients):
                    valid_recipes[i], valid_recipes[i +
                                                    1] = valid_recipes[i+1], valid_recipes[i]

        # Seleziona ricette diverse
        for recipe in valid_recipes:
            # Estrai il tipo di ricetta dal nome (es. "Pasta", "Insalata", ecc.)
            recipe_type = recipe.name.split()[0].lower()

            # Trova l'ingrediente principale (quello con il maggior contributo di CHO)
            main_ingredient = ""
            if recipe.ingredients:
                main_ingredient = max(
                    recipe.ingredients, key=lambda ing: ing.cho_contribution).name

            # Se abbiamo già una ricetta di questo tipo o con questo ingrediente principale, salta
            if (recipe_type in recipe_types) or (main_ingredient in main_ingredients and main_ingredient):
                continue

            # Altrimenti aggiungi questa ricetta
            recipe_types.add(recipe_type)
            if main_ingredient:
                main_ingredients.add(main_ingredient)
            verified_recipes.append(recipe)

            # Se abbiamo raggiunto il numero richiesto, termina
            if len(verified_recipes) >= exact_recipes_required:
                break

        # Se non abbiamo abbastanza ricette diverse, prendi le migliori comunque
        if len(verified_recipes) < exact_recipes_required:
            for recipe in valid_recipes:
                if recipe not in verified_recipes:
                    verified_recipes.append(recipe)
                    if len(verified_recipes) >= exact_recipes_required:
                        break

        print(
            f"Selezionate {len(verified_recipes)} ricette ottimali su {len(valid_recipes)} valide.")

        # Stampa dettagli delle ricette selezionate
        for i, recipe in enumerate(verified_recipes):
            print(f"Ricetta {i+1}: '{recipe.name}' - CHO: {recipe.total_cho}g (deviazione: {abs(recipe.total_cho - target_cho):.1f}g), Ingredienti: {len(recipe.ingredients)}")

    state['final_verified_recipes'] = verified_recipes

    # Aggiungi messaggio di errore se non troviamo abbastanza ricette alla fine
    if len(verified_recipes) < exact_recipes_required:
        print(
            f"Attenzione: Trovate solo {len(verified_recipes)} ricette valide, meno delle {exact_recipes_required} richieste.")
        state['error_message'] = f"Non sono state trovate abbastanza ricette ({len(verified_recipes)}/{exact_recipes_required}) che soddisfano tutti i criteri dopo la verifica finale."

    return state
