import os
import base64
from functools import lru_cache


from model_schema import GraphState


@lru_cache(maxsize=8)
def get_base64_encoded_image(image_path):
    """
    Restituisce l'immagine codificata in base64 come stringa.
    Utilizza cache per evitare di leggere ripetutamente la stessa immagine.
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Errore durante la codifica dell'immagine {image_path}: {e}")
        return None


# Dizionario di emoji fallback definito una sola volta a livello di modulo
FALLBACK_EMOJIS = {
    "vegan": '🌱',
    "vegetarian": '🥗',
    "gluten_free": '🌾',
    "lactose_free": '🥛',
}


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

    # Risolvi il percorso della cartella static dalla ROOT del progetto
    current_dir = os.path.dirname(os.path.abspath(__file__))  # agents/
    root_dir = os.path.dirname(current_dir)  # directory principale
    static_folder = os.path.join(root_dir, "static")

    # Verifica se la cartella static esiste
    if not os.path.exists(static_folder):
        print(f"ATTENZIONE: La cartella static non esiste in {static_folder}")

    # Percorsi delle immagini e loro configurazione
    icons_config = {
        "vegan": {"path": os.path.join(static_folder, "vegan.png"), "width": 24},
        "vegetarian": {"path": os.path.join(static_folder, "vegetarian.png"), "width": 24},
        "gluten_free": {"path": os.path.join(static_folder, "gluten_free.png"), "width": 24},
        "lactose_free": {"path": os.path.join(static_folder, "lactose_free.png"), "width": 24},
    }

    # Genera HTML per le icone
    img_dict = {}
    for key, config in icons_config.items():
        path = config["path"]
        width = config["width"]

        if os.path.exists(path):
            base64_image = get_base64_encoded_image(path)
            if base64_image:
                img_dict[key] = f'<img src="data:image/png;base64,{base64_image}" width="{width}" style="margin-right: 5px; vertical-align: middle;">'
            else:
                img_dict[key] = FALLBACK_EMOJIS.get(key, '⚠️')
        else:
            img_dict[key] = FALLBACK_EMOJIS.get(key, '⚠️')

    # Funzione helper per creare righe di caratteristiche
    def format_feature_row(features):
        if not features:
            return ""
        row = '<div style="display: flex; align-items: center;">'
        row += "Caratteristiche: "
        row += ", ".join(features)
        row += '</div><br>'
        return row

    # Costruisci stringa preferenze per i messaggi
    prefs_list = []
    if preferences.vegan:
        prefs_list.append(f"{img_dict['vegan']} Vegano")
    elif preferences.vegetarian:
        # Mostra solo se non vegano (perché vegano implica vegetariano)
        prefs_list.append(f"{img_dict['vegetarian']} Vegetariano")
    if preferences.gluten_free:
        prefs_list.append(f"{img_dict['gluten_free']} Senza Glutine")
    if preferences.lactose_free:
        prefs_list.append(f"{img_dict['lactose_free']} Senza Lattosio")
    prefs_string = ", ".join(
        prefs_list) if prefs_list else "Nessuna preferenza specifica"

    # Funzione helper per creare la sezione degli ingredienti
    def format_ingredients_section(ingredients):
        section = "<h3>Ingredienti</h3><ul>"
        for ing in ingredients:
            section += f"<li><b>{ing.name}:</b> {ing.quantity_g:.1f}g (CHO: {ing.cho_contribution:.1f}g"
            if hasattr(ing, 'calories_contribution') and ing.calories_contribution is not None:
                section += f", Cal: {ing.calories_contribution:.1f} kcal"
            if hasattr(ing, 'protein_contribution_g') and ing.protein_contribution_g is not None:
                section += f", Prot: {ing.protein_contribution_g:.1f}g"
            if hasattr(ing, 'fat_contribution_g') and ing.fat_contribution_g is not None:
                section += f", Grassi: {ing.fat_contribution_g:.1f}g"
            section += ")</li>"
        section += "</ul>"
        return section

    # Funzione helper per creare la sezione delle istruzioni
    def format_instructions_section(instructions):
        if not instructions:
            return ""
        section = "<h3>Preparazione</h3><ol>"
        for step in instructions:
            section += f"<li>{step}</li>"
        section += "</ol>"
        return section

    # Funzione helper per creare le caratteristiche di una ricetta
    def get_recipe_features(recipe):
        features = []
        if recipe.is_vegan:
            features.append(f"{img_dict['vegan']} Vegana")
        if recipe.is_vegetarian and not recipe.is_vegan:
            # Mostra solo se non vegano (per evitare ridondanza)
            features.append(f"{img_dict['vegetarian']} Vegetariana")
        if recipe.is_gluten_free:
            features.append(f"{img_dict['gluten_free']} Senza Glutine")
        if recipe.is_lactose_free:
            features.append(f"{img_dict['lactose_free']} Senza Lattosio")
        return features

    # Funzione helper per la sezione di nutrizione
    def format_nutrition_section(recipe):
        section = "<h3>Informazioni nutrizionali</h3><ul>"
        section += f"<li><b>CHO Totali:</b> {recipe.total_cho:.1f}g</li>"
        if recipe.total_calories is not None:
            section += f"<li><b>Calorie Totali:</b> {recipe.total_calories:.1f} kcal</li>"
        if recipe.total_protein_g is not None:
            section += f"<li><b>Proteine Totali:</b> {recipe.total_protein_g:.1f}g</li>"
        if recipe.total_fat_g is not None:
            section += f"<li><b>Grassi Totali:</b> {recipe.total_fat_g:.1f}g</li>"
        if recipe.total_fiber_g is not None:
            section += f"<li><b>Fibre Totali:</b> {recipe.total_fiber_g:.1f}g</li>"
        section += "</ul>"
        return section

    # Funzione helper per formattare una singola ricetta
    def format_recipe(recipe, index):
        output = f"<h2>{index}. {recipe.name}</h2>"

        if recipe.description:
            output += f"<p>{recipe.description}</p>"

        output += format_nutrition_section(recipe)

        recipe_features = get_recipe_features(recipe)
        output += format_feature_row(recipe_features)

        output += format_ingredients_section(recipe.ingredients)
        output += format_instructions_section(recipe.instructions)

        output += "<hr>"
        return output

    # Costruisci l'output formattato
    output_string = ""

    if final_recipes and len(final_recipes) >= min_recipes_required:
        output_string += f"<h1>Ricette personalizzate</h1><p>Ecco {len(final_recipes)} proposte di ricette che soddisfano i tuoi criteri (Target CHO: ~{preferences.target_cho:.1f}g, {prefs_string}):</p><hr>"

        for i, recipe in enumerate(final_recipes):
            output_string += format_recipe(recipe, i+1)

        output_string += "<h3>Suggerimenti</h3><p>Puoi modificare il target di carboidrati o le restrizioni dietetiche per ottenere ricette diverse.</p>"

    elif final_recipes:  # Trovate alcune ricette ma meno del minimo richiesto
        output_string += f"<h1>Risultati parziali</h1><p>Spiacente, sono state trovate solo {len(final_recipes)} ricette che soddisfano tutti i tuoi criteri (Target CHO: ~{preferences.target_cho:.1f}g, {prefs_string}) invece delle {min_recipes_required} richieste:</p><hr>"

        for i, recipe in enumerate(final_recipes):
            output_string += format_recipe(recipe, i+1)

        output_string += "<h3>Suggerimenti</h3><p>Potresti provare a modificare leggermente il target di carboidrati o le restrizioni dietetiche per ottenere più opzioni.</p>"

    else:  # Nessuna ricetta trovata o errore grave precedente
        output_string += f"<h1>Nessuna ricetta trovata</h1><p>Spiacente, non è stato possibile trovare nessuna ricetta che soddisfi tutti i tuoi criteri (Target CHO: ~{preferences.target_cho:.1f}g, {prefs_string}) entro la tolleranza richiesta.</p><hr>"

        if error_message:
            output_string += f"<p><b>Dettaglio problema:</b> {error_message}</p>"
        else:
            output_string += "<p><b>Possibile causa:</b> nessuna ricetta adatta è stata generata che rispetti tutti i vincoli specificati.</p>"

        output_string += "<h3>Suggerimenti</h3><p>Potresti provare a:</p><ul>"
        output_string += "<li>Modificare leggermente il target di carboidrati</li>"
        output_string += "<li>Rimuovere alcune delle restrizioni dietetiche, se possibile</li>"
        output_string += "<li>Verificare che i dataset degli ingredienti siano aggiornati e completi</li></ul>"

    new_state['final_output'] = output_string
    print(
        f"--- Formattazione completata ({len(output_string)} caratteri generati) ---")

    return new_state
