# agents/formatter_agent.py

from typing import List, Dict, Any, Union
from model_schema import GraphState, FinalRecipeOption, UserPreferences


def format_output_agent(state: GraphState) -> GraphState:
    """
    Node Function: Formatta l'output finale per l'utente in Markdown.
    Implementato con logica Python, non usa LLM.
    """
    print("--- ESECUZIONE NODO: Formattazione Output Finale ---")

    # Crea una copia profonda dello stato invece di modificarlo direttamente
    from copy import deepcopy
    new_state = deepcopy(state)

    final_recipes = state.get('final_verified_recipes', [])
    preferences = state['user_preferences']
    error_message = state.get('error_message')
    min_recipes_required = 3

    output_string = ""

    # Costruisci stringa preferenze per i messaggi
    prefs_list = []
    if preferences.vegan:
        prefs_list.append("Vegano")
    elif preferences.vegetarian:
        prefs_list.append("Vegetariano")  # Mostra solo se non vegano
    if preferences.gluten_free:
        prefs_list.append("Senza Glutine")
    if preferences.lactose_free:
        prefs_list.append("Senza Lattosio")
    prefs_string = ", ".join(
        prefs_list) if prefs_list else "Nessuna preferenza specifica"

    # Formatta l'output in base ai risultati
    if final_recipes and len(final_recipes) >= min_recipes_required:
        output_string += f"Ecco {len(final_recipes)} proposte di ricette che soddisfano i tuoi criteri (Target CHO: ~{preferences.target_cho:.1f}g, {prefs_string}):\n\n"

        for i, recipe in enumerate(final_recipes):
            output_string += f"**{i+1}. {recipe.name}**\n"
            output_string += f"* **CHO Totali:** {recipe.total_cho:.1f}g\n"

            # Costruisci stringa caratteristiche dietetiche della ricetta
            recipe_flags = []
            if recipe.is_vegan:
                recipe_flags.append("Vegana")
            elif recipe.is_vegetarian:
                recipe_flags.append("Vegetariana")
            if recipe.is_gluten_free:
                recipe_flags.append("Senza Glutine")
            if recipe.is_lactose_free:
                recipe_flags.append("Senza Lattosio")
            flags_string = ", ".join(
                recipe_flags) if recipe_flags else "Standard"
            output_string += f"* **Caratteristiche:** {flags_string}\n"

            output_string += "* **Ingredienti:**\n"
            for ing in recipe.ingredients:
                output_string += f"    * {ing.name}: {ing.quantity_g:.1f}g (CHO: {ing.cho_contribution:.1f}g)\n"
            output_string += "\n"  # Spazio tra le ricette

    elif final_recipes:  # Trovate alcune ricette ma meno del minimo richiesto
        output_string += f"Spiacente, sono state trovate solo {len(final_recipes)} ricette che soddisfano tutti i tuoi criteri (Target CHO: ~{preferences.target_cho:.1f}g, {prefs_string}) invece delle {min_recipes_required} richieste:\n\n"
        # Lista comunque quelle trovate
        for i, recipe in enumerate(final_recipes):
            output_string += f"**{i+1}. {recipe.name}**\n"
            output_string += f"* **CHO Totali:** {recipe.total_cho:.1f}g\n"
            recipe_flags = []
            if recipe.is_vegan:
                recipe_flags.append("Vegana")
            elif recipe.is_vegetarian:
                recipe_flags.append("Vegetariana")
            if recipe.is_gluten_free:
                recipe_flags.append("Senza Glutine")
            if recipe.is_lactose_free:
                recipe_flags.append("Senza Lattosio")
            flags_string = ", ".join(
                recipe_flags) if recipe_flags else "Standard"
            output_string += f"* **Caratteristiche:** {flags_string}\n"
            output_string += "* **Ingredienti:**\n"
            for ing in recipe.ingredients:
                output_string += f"    * {ing.name}: {ing.quantity_g:.1f}g (CHO: {ing.cho_contribution:.1f}g)\n"
            output_string += "\n"
        output_string += "\nPotresti provare a modificare leggermente il target di carboidrati o le restrizioni dietetiche."

    else:  # Nessuna ricetta trovata o errore grave precedente
        output_string += f"Spiacente, non è stato possibile trovare nessuna ricetta che soddisfi tutti i tuoi criteri (Target CHO: ~{preferences.target_cho:.1f}g, {prefs_string}) entro la tolleranza richiesta.\n"
        if error_message and "Nessuna ricetta iniziale trovata" in error_message:
            output_string += "Non ci sono ricette nel database che corrispondano alle tue preferenze dietetiche iniziali.\n"
        elif error_message:
            # Riporta l'ultimo errore significativo
            output_string += f"Dettaglio problema: {error_message}\n"
        else:
            output_string += "Possibili cause: nessuna ricetta adatta nel database, oppure l'adattamento non è stato possibile in modo sensato.\n"

        output_string += "\nPotresti provare a:\n"
        output_string += "* Modificare leggermente il target di carboidrati.\n"
        output_string += "* Rimuovere alcune delle restrizioni dietetiche, se possibile.\n"
        output_string += "* Verificare che i dataset 'ingredients.csv' e 'recipes.csv' siano presenti e formattati correttamente nella cartella 'data'."

    new_state['final_output'] = output_string
    print(
        f"--- Formattazione completata ({len(output_string)} caratteri generati) ---")

    # Assicurati che l'attributo final_output sia visibile quando si stampa lo stato
    print(f"Output finale impostato con {len(output_string)} caratteri")

    return new_state
