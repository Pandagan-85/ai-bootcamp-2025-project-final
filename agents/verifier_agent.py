# agents/verifier_agent.py

from typing import List
from model_schema import GraphState, FinalRecipeOption, UserPreferences
# Usa la versione per FinalRecipeOption
from utils import check_final_recipe_dietary_match


def verifier_agent(state: GraphState) -> GraphState:
    """
    Node Function: Verifica le ricette processate rispetto ai criteri finali.
    Implementato con logica Python, non usa LLM.
    """
    print("--- ESECUZIONE NODO: Verifica Finale Ricette ---")
    preferences: UserPreferences = state['user_preferences']
    processed_recipes: List[FinalRecipeOption] = state.get(
        'processed_recipes', [])
    target_cho = preferences.target_cho
    cho_tolerance = 6.0  # Tolleranza fissa +/- 6g
    exact_recipes_required = 3  # Numero esatto di ricette da fornire

    if not processed_recipes:
        print("Nessuna ricetta processata da verificare.")
        state['final_verified_recipes'] = []
        # Potresti voler impostare un messaggio d'errore specifico qui
        if not state.get('error_message'):
            state['error_message'] = "Nessuna ricetta è stata generata o modificata con successo."
        return state

    valid_recipes: List[FinalRecipeOption] = []
    min_cho = target_cho - cho_tolerance
    max_cho = target_cho + cho_tolerance

    print(
        f"Verifica di {len(processed_recipes)} ricette processate. Target CHO: {target_cho}g (Range: {min_cho}g - {max_cho}g)")

    for recipe in processed_recipes:
        # 1. Verifica CHO
        cho_ok = min_cho <= recipe.total_cho <= max_cho
        if not cho_ok:
            print(
                f"Ricetta '{recipe.name}' SCARTATA: CHO ({recipe.total_cho}g) fuori range ({min_cho}-{max_cho}g).")
            continue

        # 2. Verifica Dieta
        diet_ok = check_final_recipe_dietary_match(recipe, preferences)
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

        # 3. Verifica Equilibrio (Molto Semplice per MVP)
        # Scarta ricette con un solo ingrediente se non è un pasto completo ovvio (es. frutta)
        # O se un singolo ingrediente fornisce >95% dei CHO (indice di squilibrio)
        is_unbalanced = False
        # Esempio: singolo ingrediente > 20 CHO è sospetto
        if len(recipe.ingredients) == 1 and recipe.total_cho > 20:
            # Potresti aggiungere eccezioni qui (es. frutta come spuntino)
            print(
                f"Ricetta '{recipe.name}' SCARTATA: Potenzialmente sbilanciata (singolo ingrediente).")
            is_unbalanced = True

        # Controllo se un ingrediente domina troppo i CHO
        if not is_unbalanced and recipe.total_cho > 0:  # Evita divisione per zero
            for ing in recipe.ingredients:
                if (ing.cho_contribution / recipe.total_cho) > 0.95 and len(recipe.ingredients) > 1:
                    print(
                        f"Ricetta '{recipe.name}' SCARTATA: Potenzialmente sbilanciata (un ingrediente fornisce >95% CHO).")
                    is_unbalanced = True
                    break  # Basta un ingrediente dominante

        if is_unbalanced:
            continue

        # Se tutti i check sono OK, aggiungi alla lista di ricette valide
        print(
            f"Ricetta '{recipe.name}' VERIFICATA (CHO: {recipe.total_cho}g, Dieta OK).")
        valid_recipes.append(recipe)

    print(
        f"Ricette che hanno passato tutte le verifiche: {len(valid_recipes)}")

    # Selezione delle migliori ricette (massimo 3)
    verified_recipes = []

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

        # Prendi al massimo 3 ricette
        verified_recipes = valid_recipes[:min(
            exact_recipes_required, len(valid_recipes))]

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
