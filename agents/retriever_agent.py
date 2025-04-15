# agents/retriever_agent.py

from typing import List
from model_schema import Recipe, UserPreferences, GraphState
from utils import check_dietary_match  # Importa la funzione di check


def initial_recipe_retriever(state: GraphState) -> GraphState:
    """
    Node Function: Filtra le ricette iniziali basandosi sulle preferenze dietetiche.
    Non usa LLM, è un semplice filtro.
    """
    print("--- ESECUZIONE NODO: Recupero Ricette Iniziali ---")
    preferences: UserPreferences = state['user_preferences']
    # Recupera tutte le ricette caricate (da aggiungere allo stato iniziale)
    all_recipes: List[Recipe] = state.get('initial_recipes', [])

    if not all_recipes:
        print("Attenzione: Nessuna ricetta trovata nel dataset caricato.")
        state['initial_recipes'] = []
        state['error_message'] = "Nessuna ricetta disponibile nel database."
        return state

    print(f"Filtraggio di {len(all_recipes)} ricette totali per preferenze: Vegan={preferences.vegan}, Veg={preferences.vegetarian}, GF={preferences.gluten_free}, LF={preferences.lactose_free}")

    # Filtra le ricette che soddisfano i criteri dietetici
    matching_recipes = [
        recipe for recipe in all_recipes
        if check_dietary_match(recipe, preferences)
    ]

    print(
        f"Trovate {len(matching_recipes)} ricette iniziali compatibili con le preferenze dietetiche.")

    # Aggiorna lo stato con le ricette filtrate
    state['initial_recipes'] = matching_recipes
    if not matching_recipes:
        state['error_message'] = "Nessuna ricetta iniziale trovata compatibile con le preferenze dietetiche."

    return state

# Nota: Questa funzione verrà usata come un nodo nel grafo LangGraph.
# Riceve lo stato, lo modifica e lo restituisce.
