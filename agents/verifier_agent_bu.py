"""
Agente di verifica e ottimizzazione delle ricette generate.

Questo modulo implementa il nodo 'verifier_agent' del workflow, responsabile
dell'analisi, filtraggio e aggiustamento delle ricette proposte dal generatore.
NON esegue più il bilanciamento della distribuzione dei CHO tra gli ingredienti.

Le operazioni principali svolte includono:
1.  **Validazione Ingredienti:** Verifica che tutti gli ingredienti di una ricetta
    siano presenti nel database (`ingredient_data`). Le ricette con ingredienti
    sconosciuti vengono scartate (`validate_recipe_ingredients`).
2.  **Aggiustamento CHO:** Tenta di modificare le quantità degli ingredienti
    (`adjust_recipe_cho`) per portare il contenuto totale di carboidrati (CHO)
    della ricetta entro un range accettabile rispetto al target dell'utente.
    Questo passaggio viene applicato alle ricette che non rientrano già in un
    range di tolleranza predefinito rispetto al loro valore originale.
3.  **Verifica Finale Multi-criterio:** Applica una serie di controlli finali
    sulle ricette (originali o aggiustate) per scartare quelle che non
    soddisfano i requisiti stringenti:
    - Range di CHO finale (target ± tolleranza).
    - Conformità alle preferenze dietetiche specificate dall'utente (vegan, vegetarian, gluten-free, lactose-free).
    - Criteri di 'qualità' (es. numero minimo di ingredienti > 2, quantità massime per singolo ingrediente < 300g).
    - Controllo (meno restrittivo) su ingredienti singoli con contributo CHO eccessivamente dominante (>98%).

L'agente riceve le ricette generate (`generated_recipes`), applica queste fasi
di verifica e selezione, e restituisce una lista finale (`final_verified_recipes`)
contenente le ricette che hanno superato tutti i controlli, pronte per la
formattazione finale. Gestisce anche la diversificazione delle ricette selezionate
e imposta messaggi di errore se non vengono trovate abbastanza ricette valide.
"""
from typing import List, Dict
from copy import deepcopy
from model_schema import GraphState, FinalRecipeOption, UserPreferences, RecipeIngredient, IngredientInfo

from utils import calculate_ingredient_cho_contribution


def adjust_recipe_cho(recipe: FinalRecipeOption, target_cho: float, ingredient_data: Dict[str, IngredientInfo]) -> FinalRecipeOption:
    """
    Aggiusta le quantità degli ingredienti per raggiungere il target CHO desiderato,
    con limiti di scaling più aggressivi e logica graduale.
    """
    # Verifica iniziale e copia (come prima)
    if recipe.total_cho is None:
        print(
            f"Warning adjust_cho: Ricetta '{recipe.name}' ha total_cho=None. Impossibile aggiustare.")
        return recipe  # Non fare nulla se manca il CHO iniziale

    # Se siamo già molto vicini al target, non fare nulla
    if abs(recipe.total_cho - target_cho) < 5.0:  # Soglia di non intervento ridotta
        # print(f"Debug adjust_cho: CHO ({recipe.total_cho:.1f}g) già vicino al target ({target_cho:.1f}g). Nessun aggiustamento.")
        return recipe

    adjusted_recipe = deepcopy(recipe)
    initial_cho = adjusted_recipe.total_cho  # Salva CHO iniziale per confronto

    # Identifica ingredienti ricchi di CHO (logica come prima)
    cho_rich_ingredients = []
    # ... (la tua logica esistente per popolare cho_rich_ingredients) ...
    # Esempio di logica (assicurati che sia robusta):
    if adjusted_recipe.total_cho > 0:  # Evita divisione per zero
        for ing in adjusted_recipe.ingredients:
            # Considera ingredienti con almeno 5g CHO/100g
            if ing.name in ingredient_data and ingredient_data[ing.name].cho_per_100g > 5:
                # Usa contributo già calcolato se disponibile e valido
                cho_contribution = ing.cho_contribution if ing.cho_contribution is not None and ing.cho_contribution > 0 else 0
                # Considera "ricco" se contribuisce >5% o se CHO totale è basso
                if cho_contribution > 0 and (adjusted_recipe.total_cho < 30 or (cho_contribution / adjusted_recipe.total_cho) > 0.05):
                    cho_rich_ingredients.append(ing)

    # Se non trovati, prova fallback (come prima, es. > 5g CHO/100g)
    if not cho_rich_ingredients:
        potential_ingredients = [
            ing for ing in adjusted_recipe.ingredients
            if ing.name in ingredient_data and ingredient_data[ing.name].cho_per_100g > 5
        ]
        if potential_ingredients:
            cho_rich_ingredients = potential_ingredients

    if not cho_rich_ingredients:
        print(
            f"Debug adjust_cho: No CHO-rich ingredients identified for '{recipe.name}' to scale.")
        return recipe  # Non possiamo aggiustare se non sappiamo cosa scalare

    # Ordina per contributo (come prima)
    cho_rich_ingredients.sort(
        key=lambda x: x.cho_contribution if x.cho_contribution is not None else 0, reverse=True)
    # print(f"Debug adjust_cho: Identified {len(cho_rich_ingredients)} CHO-rich ingredients for '{recipe.name}': {[ing.name for ing in cho_rich_ingredients]}")

    # Calcola differenza CHO (come prima)
    cho_diff = target_cho - adjusted_recipe.total_cho
    print(
        f"Debug adjust_cho: Recipe '{recipe.name}' - Initial CHO: {initial_cho:.1f}g, Target: {target_cho:.1f}g, Diff: {cho_diff:.1f}g")

    # --- INIZIO BLOCCO MODIFICATO: Calcolo e Applicazione Scaling Factor ---
    scaling_factor = 1.0  # Default a nessun cambiamento

    if adjusted_recipe.total_cho > 0:  # Evita divisione per zero
        # Calcolo base del fattore di scala ideale
        ideal_scaling_factor = target_cho / adjusted_recipe.total_cho

        # Definisci i limiti minimi e massimi per lo scaling
        min_scale = 0.3  # Limite inferiore più permissivo (riduzione maggiore)
        # Limite superiore MOLTO aggressivo per differenze > 100g
        max_scale_very_high_diff = 7.0
        max_scale_high_diff = 6.0  # Limite superiore aggressivo per differenze > 60g
        max_scale_med_diff = 4.5  # Limite superiore moderato per differenze > 30g
        max_scale_low_diff = 3.0  # Limite superiore leggero per differenze < 30g

        # Applica i limiti basati sulla differenza CHO e sulla direzione (aumentare/ridurre)
        if cho_diff < 0:  # Dobbiamo RIDURRE i CHO (total_cho > target_cho)
            # Determina limite inferiore in base all'entità della differenza
            if abs(cho_diff) > 60:
                effective_min_scale = min_scale  # Molto basso per grandi riduzioni
            elif abs(cho_diff) > 30:
                effective_min_scale = 0.4
            else:
                effective_min_scale = 0.5  # Riduzione meno aggressiva per differenze minori
            # Applica il limite inferiore
            scaling_factor = max(effective_min_scale, ideal_scaling_factor)
            # Assicurati che non superi 1.0 (non vogliamo aumentare se dobbiamo ridurre)
            scaling_factor = min(scaling_factor, 1.0)
            print(
                f"DEBUG adjust_cho (Riduzione): Diff={cho_diff:.1f}g, IdealScale={ideal_scaling_factor:.2f}, EffectiveMinScale={effective_min_scale:.2f}, FinalScale={scaling_factor:.2f}")

        elif cho_diff > 0:  # Dobbiamo AUMENTARE i CHO (total_cho < target_cho)
            # Determina il limite massimo in base all'entità della differenza
            if abs(cho_diff) > 100:  # Differenza molto grande
                effective_max_scale = max_scale_very_high_diff
                print(
                    f"DEBUG adjust_cho (Aumento >100g): Applico max_scale {effective_max_scale}")
            elif abs(cho_diff) > 60:  # Differenza grande
                effective_max_scale = max_scale_high_diff
                print(
                    f"DEBUG adjust_cho (Aumento >60g): Applico max_scale {effective_max_scale}")
            elif abs(cho_diff) > 30:  # Differenza media
                effective_max_scale = max_scale_med_diff
                print(
                    f"DEBUG adjust_cho (Aumento >30g): Applico max_scale {effective_max_scale}")
            else:  # Differenza minore
                effective_max_scale = max_scale_low_diff
                print(
                    f"DEBUG adjust_cho (Aumento <30g): Applico max_scale {effective_max_scale}")

            # Applica il limite superiore
            scaling_factor = min(effective_max_scale, ideal_scaling_factor)
            # Assicurati che sia almeno 1.0 (non vogliamo ridurre se dobbiamo aumentare)
            scaling_factor = max(scaling_factor, 1.0)
            print(
                f"DEBUG adjust_cho (Aumento): Diff={cho_diff:.1f}g, IdealScale={ideal_scaling_factor:.2f}, EffectiveMaxScale={effective_max_scale:.2f}, FinalScale={scaling_factor:.2f}")

        else:  # cho_diff è 0 (improbabile data la verifica iniziale, ma per sicurezza)
            scaling_factor = 1.0
            print("DEBUG adjust_cho: Diff=0, nessun scaling necessario.")

        # Applica lo scaling agli ingredienti ricchi di CHO
        new_recipe_ingredients_list = []
        adjusted_ingredient_names = []  # Per log
        for ing in adjusted_recipe.ingredients:
            new_quantity = ing.quantity_g
            # Applica scaling solo se è ricco di CHO e il nome è valido nel DB
            if ing in cho_rich_ingredients and ing.name in ingredient_data:
                original_quantity = ing.quantity_g
                scaled_quantity = original_quantity * scaling_factor

                # Applica limiti MIN/MAX ASSOLUTI alla quantità finale
                # Considera se 300g è sufficiente o serve aumentarlo a 350g
                MAX_SINGLE_INGREDIENT_G = 300.0
                MIN_SINGLE_INGREDIENT_G = 5.0  # Evita quantità troppo piccole
                new_quantity = max(MIN_SINGLE_INGREDIENT_G, min(
                    MAX_SINGLE_INGREDIENT_G, scaled_quantity))

                # Log se la quantità è cambiata significativamente
                if abs(new_quantity - original_quantity) > 0.1:
                    adjusted_ingredient_names.append(
                        f"{ing.name} ({original_quantity:.1f}g -> {new_quantity:.1f}g)")

            new_recipe_ingredients_list.append(
                RecipeIngredient(name=ing.name, quantity_g=round(
                    new_quantity, 1))  # Arrotonda a 1 decimale
            )

        if adjusted_ingredient_names:
            print(
                f"DEBUG adjust_cho: Ingredienti scalati ({len(adjusted_ingredient_names)}): {'; '.join(adjusted_ingredient_names)}")
        elif scaling_factor != 1.0:
            print(
                f"DEBUG adjust_cho: Scaling factor ({scaling_factor:.2f}) applicato, ma nessun ingrediente CHO-rich modificato significativamente.")
        else:
            print(f"DEBUG adjust_cho: Nessun scaling applicato (scaling_factor=1.0).")

        # Ricalcola tutto dopo scaling (come prima)
        updated_ingredients = calculate_ingredient_cho_contribution(
            new_recipe_ingredients_list, ingredient_data)
        adjusted_recipe.ingredients = updated_ingredients
        adjusted_recipe.total_cho = round(sum(
            ing.cho_contribution for ing in updated_ingredients if ing.cho_contribution is not None), 2)
        print(
            f"DEBUG adjust_cho: CHO dopo scaling: {adjusted_recipe.total_cho:.1f}g")

    else:  # Caso adjusted_recipe.total_cho <= 0
        print(
            f"Warning adjust_cho: CHO iniziale non positivo ({adjusted_recipe.total_cho}g) per '{recipe.name}'. Impossibile scalare.")
        # Non fare nulla se CHO iniziale non è valido per lo scaling
    # --- FINE BLOCCO MODIFICATO ---

    # --- INIZIO BLOCCO AGGIUSTAMENTO FINE (Mantieni la tua logica esistente) ---
    # Aggiustamento fine (logica come prima, assicurati usi nomi DB corretti)
    # Ricalcola la differenza *dopo* lo scaling
    current_cho_diff_after_scaling = target_cho - adjusted_recipe.total_cho
    fine_tuning_threshold = 5.0  # Soglia per attivare l'aggiustamento fine

    if abs(current_cho_diff_after_scaling) > fine_tuning_threshold and cho_rich_ingredients:
        print(
            f"DEBUG adjust_cho: Tentativo aggiustamento fine (Diff: {current_cho_diff_after_scaling:.1f}g)")
        # Trova l'ingrediente principale (già ordinati)
        main_ingredient_obj = cho_rich_ingredients[0]
        main_ing_name = main_ingredient_obj.name

        if main_ing_name in ingredient_data and ingredient_data[main_ing_name].cho_per_100g is not None and ingredient_data[main_ing_name].cho_per_100g > 0:
            cho_per_100g = ingredient_data[main_ing_name].cho_per_100g
            # Calcola la modifica in grammi necessaria per l'ingrediente principale
            gram_diff = (current_cho_diff_after_scaling / cho_per_100g) * 100.0
            print(
                f"DEBUG adjust_cho (Fine): Modifica '{main_ing_name}' di {gram_diff:.1f}g (CHO/100g: {cho_per_100g})")

            # Modifica solo l'ingrediente principale nella lista
            fine_tuned_ingredients_list = []
            found = False
            main_ing_original_qty = 0
            main_ing_final_qty = 0
            for ing in adjusted_recipe.ingredients:
                new_quantity = ing.quantity_g
                if ing.name == main_ing_name and not found:
                    main_ing_original_qty = ing.quantity_g
                    adjusted_quantity = main_ing_original_qty + gram_diff
                    # Applica limiti MIN/MAX anche qui
                    MAX_SINGLE_INGREDIENT_G = 300.0
                    # Minimo più alto per aggiustamento fine? Valuta.
                    MIN_SINGLE_INGREDIENT_G = 10.0
                    new_quantity = max(MIN_SINGLE_INGREDIENT_G, min(
                        MAX_SINGLE_INGREDIENT_G, adjusted_quantity))
                    main_ing_final_qty = new_quantity
                    found = True  # Modifica solo la prima occorrenza
                fine_tuned_ingredients_list.append(
                    RecipeIngredient(name=ing.name, quantity_g=round(
                        new_quantity, 1))  # Arrotonda
                )

            if found:
                print(
                    f"DEBUG adjust_cho (Fine): Quantità '{main_ing_name}' cambiata da {main_ing_original_qty:.1f}g a {main_ing_final_qty:.1f}g.")
                # Ricalcola di nuovo
                updated_ingredients_final = calculate_ingredient_cho_contribution(
                    fine_tuned_ingredients_list, ingredient_data)
                adjusted_recipe.ingredients = updated_ingredients_final
                adjusted_recipe.total_cho = round(sum(
                    ing.cho_contribution for ing in updated_ingredients_final if ing.cho_contribution is not None), 2)
                print(
                    f"DEBUG adjust_cho: CHO dopo aggiustamento fine: {adjusted_recipe.total_cho:.1f}g")
            else:
                print(
                    f"DEBUG adjust_cho (Fine): Ingrediente principale '{main_ing_name}' non trovato/modificato.")
        else:
            print(
                f"DEBUG adjust_cho (Fine): Impossibile aggiustare '{main_ing_name}' (manca info CHO o CHO <= 0).")

    # --- FINE BLOCCO AGGIUSTAMENTO FINE ---

    # --- AGGIORNAMENTO NOME/DESCRIZIONE (Come prima) ---
    # Aggiungi "(Aggiustata)" al nome se CHO è cambiato significativamente
    if abs(initial_cho - adjusted_recipe.total_cho) > 1.0:  # Se c'è stata una modifica > 1g
        if "Aggiustata" not in adjusted_recipe.name:  # Evita doppi suffissi
            adjusted_recipe.name = f"{adjusted_recipe.name} (Aggiustata)"
        # Aggiorna descrizione (come prima)
        if adjusted_recipe.description and "aggiustate" not in adjusted_recipe.description.lower():
            adjusted_recipe.description += " Quantità aggiustate per target CHO."

    return adjusted_recipe


def verifier_agent(state: GraphState) -> GraphState:
    """
    Node Function: Verifica le ricette generate dopo aggiustamento CHO.
    """
    print("--- ESECUZIONE NODO: Verifica Finale Ricette ---")
    preferences: UserPreferences = state['user_preferences']
    # Riceve ricette GIA' validate a livello di ingredienti dal generator
    generated_recipes: List[FinalRecipeOption] = state.get(
        'generated_recipes', [])
    # Estrai il dizionario ingredienti dal pacchetto
    ingredient_data = state.get('available_ingredients_data', {})

    if not ingredient_data:
        state['error_message'] = "Dati ingredienti mancanti per la verifica finale."
        print(f"Errore: {state['error_message']}")
        state['final_verified_recipes'] = []
        return state

    target_cho = preferences.target_cho
    cho_tolerance = 25.0
    exact_recipes_required = 2  # Numero desiderato
    # Soglia ingrediente dominante (per scarto finale)
    max_cho_dominance = 0.99

    if not generated_recipes:
        print("Nessuna ricetta valida generata da verificare.")
        # Imposta errore se non già presente
        if not state.get('error_message'):
            state['error_message'] = "Nessuna ricetta valida è stata generata o mappata correttamente."
        state['final_verified_recipes'] = []
        return state

    # Definisci range CHO
    min_cho = target_cho - cho_tolerance
    max_cho = target_cho + cho_tolerance
    print(
        f"Verifica di {len(generated_recipes)} ricette. Target CHO: {target_cho}g (Range: {min_cho}g - {max_cho}g)")

    # RIMOSSA VALIDAZIONE INIZIALE INGREDIENTI QUI

    # Fase 1: Prepara per aggiustamento (Ex-bilanciamento)
    recipes_to_adjust = generated_recipes  # Usa direttamente quelle generate
    print(
        f"Fase 1: Preparazione di {len(recipes_to_adjust)} ricette per aggiustamento CHO.")

    # Fase 2: Prova ad aggiustare TUTTE le ricette per avvicinarsi al target CHO
    adjusted_recipes = []
    print(f"Fase 2: Tentativo di aggiustamento CHO (Target: {target_cho}g)")
    for recipe in recipes_to_adjust:
        # Tenta di aggiustare ogni ricetta
        adjusted_recipe = adjust_recipe_cho(
            recipe, target_cho, ingredient_data)
        if adjusted_recipe.name != recipe.name:  # Se è stata modificata
            print(
                f"Ricetta '{recipe.name}' -> '{adjusted_recipe.name}' AGGIUSTATA: CHO da {recipe.total_cho:.1f}g a {adjusted_recipe.total_cho:.1f}g")
        adjusted_recipes.append(adjusted_recipe)

    # Fase 3: Verifica finale multi-criterio sulle ricette aggiustate (o originali se non modificate)
    print("Fase 3: Applicazione verifiche finali (CHO, Dieta, Qualità)")
    valid_recipes_final: List[FinalRecipeOption] = []
    for recipe in adjusted_recipes:
        # 1. Verifica Range CHO Finale
        cho_ok = min_cho <= recipe.total_cho <= max_cho
        if not cho_ok:
            print(
                f"Ricetta '{recipe.name}' SCARTATA [Fase 3]: CHO ({recipe.total_cho:.1f}g) fuori range ({min_cho:.1f}-{max_cho:.1f}g).")
            continue

        # 2. Verifica Dieta (Usa flag GIA' CORRETTI dal generator)
        diet_ok = True
        required_prefs_str = []  # Per logging
        if preferences.vegan and not recipe.is_vegan:
            diet_ok = False
            if "Vegano" not in required_prefs_str:
                required_prefs_str.append("Vegano")
        # Se non vegano richiesto, check vegetariano (evita ridondanza nel log)
        if preferences.vegetarian and not preferences.vegan and not recipe.is_vegetarian:
            diet_ok = False
            if "Vegetariano" not in required_prefs_str:
                required_prefs_str.append("Vegetariano")
        if preferences.gluten_free and not recipe.is_gluten_free:
            diet_ok = False
            if "Senza Glutine" not in required_prefs_str:
                required_prefs_str.append("Senza Glutine")
        if preferences.lactose_free and not recipe.is_lactose_free:
            diet_ok = False
            if "Senza Lattosio" not in required_prefs_str:
                required_prefs_str.append("Senza Lattosio")

        if not diet_ok:
            prefs_str = ", ".join(
                required_prefs_str) if required_prefs_str else "Standard"
            # Log più chiaro dei flag ricetta
            recipe_flags = f"V={recipe.is_vegan}, VG={recipe.is_vegetarian}, GF={recipe.is_gluten_free}, LF={recipe.is_lactose_free}"
            print(
                f"Ricetta '{recipe.name}' SCARTATA [Fase 3]: Non soddisfa preferenze dietetiche richieste ({prefs_str}). Flag ricetta: {recipe_flags}")
            continue

        # 3. Verifica Qualità/Equilibrio
        # Scarta ricette troppo semplici
        min_ingredients = 3
        if len(recipe.ingredients) < min_ingredients:
            print(
                f"Ricetta '{recipe.name}' SCARTATA [Fase 3]: Troppo semplice (solo {len(recipe.ingredients)} ingredienti, min: {min_ingredients}).")
            continue

        # Scarta ricette con quantità eccessive (controllo fatto anche in adjust, ma ricontrolla)
        max_quantity_g = 300.0
        has_excessive_ingredient = False
        for ing in recipe.ingredients:
            if ing.quantity_g > max_quantity_g:
                print(
                    f"Ricetta '{recipe.name}' SCARTATA [Fase 3]: Ingrediente '{ing.name}' con quantità eccessiva ({ing.quantity_g:.1f}g > {max_quantity_g}g).")
                has_excessive_ingredient = True
                break
        if has_excessive_ingredient:
            continue

        # Scarta ricette troppo sbilanciate (un ingrediente domina i CHO)
        is_unbalanced = False
        # Verifica solo se ci sono CHO e più di 2 ingredienti per evitare casi limite
        if recipe.total_cho > 1.0 and len(recipe.ingredients) > 2:
            for ing in recipe.ingredients:
                # Assicurati che cho_contribution sia valido prima della divisione
                if ing.cho_contribution is not None and ing.cho_contribution > 0:
                    ratio = ing.cho_contribution / recipe.total_cho
                    if ratio > max_cho_dominance:
                        print(
                            f"Ricetta '{recipe.name}' SCARTATA [Fase 3]: Sbilanciata. Ingrediente '{ing.name}' fornisce >{int(max_cho_dominance*100)}% dei CHO ({ratio:.1%}).")
                        is_unbalanced = True
                        break
        if is_unbalanced:
            continue

        # Se tutti i check sono OK
        print(
            f"Ricetta '{recipe.name}' VERIFICATA [Fase 3] (CHO: {recipe.total_cho:.1f}g, Dieta OK, Ingredienti: {len(recipe.ingredients)}).")
        valid_recipes_final.append(recipe)

    print(
        f"Ricette che hanno passato tutte le verifiche finali: {len(valid_recipes_final)}")

    # Fase 4: Selezione Finale e Ordinamento (logica come prima)
    verified_recipes_selected: List[FinalRecipeOption] = []
    recipe_types = set()
    main_ingredients_set = set()

    if valid_recipes_final:
        # Ordina per vicinanza al target CHO (come prima)
        valid_recipes_final.sort(key=lambda r: abs(r.total_cho - target_cho))
        # Logica per rompere parità (come prima)
        for i in range(len(valid_recipes_final)-1):
            if abs(valid_recipes_final[i].total_cho - target_cho) == abs(valid_recipes_final[i+1].total_cho - target_cho):
                if len(valid_recipes_final[i].ingredients) < len(valid_recipes_final[i+1].ingredients):
                    valid_recipes_final[i], valid_recipes_final[i +
                                                                1] = valid_recipes_final[i+1], valid_recipes_final[i]

        # Seleziona ricette diverse (come prima)
        for recipe in valid_recipes_final:
            recipe_type = recipe.name.split(
            )[0].lower() if recipe.name else "unknown"
            main_ingredient = ""
            if recipe.ingredients:
                # Trova ingrediente con più CHO (assicurati cho_contribution esista)
                try:
                    main_ingredient = max(
                        recipe.ingredients, key=lambda ing: ing.cho_contribution if ing.cho_contribution is not None else -1).name
                except ValueError:  # Lista ingredienti potrebbe essere vuota se filtrati tutti?
                    main_ingredient = ""

            # Check unicità tipo e ingrediente principale
            if (recipe_type in recipe_types) or (main_ingredient and main_ingredient in main_ingredients_set):
                continue

            recipe_types.add(recipe_type)
            if main_ingredient:
                main_ingredients_set.add(main_ingredient)
            verified_recipes_selected.append(recipe)
            if len(verified_recipes_selected) >= exact_recipes_required:
                break

        # Riempi se non abbastanza diverse (come prima)
        if len(verified_recipes_selected) < exact_recipes_required:
            for recipe in valid_recipes_final:
                if recipe not in verified_recipes_selected:
                    verified_recipes_selected.append(recipe)
                    if len(verified_recipes_selected) >= exact_recipes_required:
                        break

        print(
            f"Selezionate {len(verified_recipes_selected)} ricette ottimali su {len(valid_recipes_final)} valide.")
        # Stampa dettagli selezionate (come prima)
        for i, recipe in enumerate(verified_recipes_selected):
            print(f"Ricetta {i+1}: '{recipe.name}' - CHO: {recipe.total_cho:.1f}g (Dev: {abs(recipe.total_cho - target_cho):.1f}g), N.Ingr: {len(recipe.ingredients)}")

    # Aggiorna stato finale
    state['final_verified_recipes'] = verified_recipes_selected

    # Aggiungi messaggio di errore se necessario (come prima)
    if len(verified_recipes_selected) < exact_recipes_required:
        err_msg = f"Trovate solo {len(verified_recipes_selected)} ricette valide, meno delle {exact_recipes_required} richieste."
        print(f"Attenzione: {err_msg}")
        # Sovrascrivi errore precedente solo se più specifico
        state['error_message'] = state.get(
            'error_message', '') + f" ({err_msg})" if state.get('error_message') else err_msg
    # elif state.get('error_message') and len(verified_recipes_selected) > 0:
        # Rimuovi eventuali errori precedenti se abbiamo trovato ricette alla fine
        # state.pop('error_message', None) # Commentato: forse è meglio tenere errori precedenti?

    return state
