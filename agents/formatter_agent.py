"""
Agente di formattazione per il sistema di generazione ricette.

Questo modulo contiene il nodo di formattazione dell'output finale che trasforma i dati
delle ricette in una presentazione HTML strutturata e visivamente gradevole per l'utente.
Include la gestione di immagini, icone e fallback emoji per indicare le caratteristiche
delle ricette.
"""
import os
import base64
from functools import lru_cache


from model_schema import GraphState


@lru_cache(maxsize=8)
def get_base64_encoded_image(image_path):
    """
    Restituisce l'immagine codificata in base64 come stringa.

    Converte un'immagine in una stringa base64 che può essere inserita direttamente
    nell'HTML. La funzione utilizza la cache LRU per evitare di leggere e codificare
    ripetutamente la stessa immagine, migliorando le performance.

    Args:
        image_path: Percorso completo all'immagine da codificare.

    Returns:
        str: Stringa contenente l'immagine codificata in base64 o None in caso di errore.

    Note:
        - Implementata con decorator @lru_cache per memorizzare i risultati
        - Una cache di dimensione 8 è sufficiente poiché abbiamo solo poche icone
    """
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Errore durante la codifica dell'immagine {image_path}: {e}")
        return None


# Dizionario di emoji fallback definito una sola volta a livello di modulo, quando non trova le img png specifiche userà queste
FALLBACK_EMOJIS = {
    "vegan": '🌱',
    "vegetarian": '🥗',
    "gluten_free": '🌾',
    "lactose_free": '🥛',
}


def format_output_agent(state: GraphState) -> GraphState:
    """
    Node Function: Formatta l'output finale per l'utente in HTML ricco.

    Questo nodo è l'ultimo nel flusso di generazione delle ricette. Prende le ricette verificate e le formatta in una presentazione HTML strutturata con sezioni per caratteristiche, ingredienti, istruzioni e valori nutrizionali.

    Args:
        state: Lo stato corrente del grafo, contenente le ricette verificate.

    Returns:
        GraphState: Lo stato aggiornato con l'aggiunta del campo 'final_output' contenente la stringa HTML formattata.

    Note:
        - Implementato con logica Python pura, non utilizza LLM
        - Gestisce tre scenari principali:
          1. Successo (≥3 ricette trovate)
          2. Successo parziale (1-2 ricette trovate)
          3. Nessuna ricetta trovata (errore o vincoli troppo restrittivi)
        - Include gestione delle immagini con fallback a emoji
        - Organizza l'output in sezioni per una migliore leggibilità
    """
    print("--- ESECUZIONE NODO: Formattazione Output Finale ---")

    # Crea una copia profonda dello stato invece di modificarlo direttamente
    from copy import deepcopy
    new_state = deepcopy(state)

    final_recipes = state.get('final_verified_recipes', [])
    preferences = state['user_preferences']
    error_message = state.get('error_message')
    min_recipes_required = 3  # Soglia per considerare il processo completamente riuscito

    # Risolvi il percorso della cartella static dalla ROOT del progetto
    # Questo permette al codice di funzionare indipendentemente da dove viene eseguito
    current_dir = os.path.dirname(os.path.abspath(__file__))  # agents/
    root_dir = os.path.dirname(current_dir)  # directory principale
    static_folder = os.path.join(root_dir, "static")

    # Verifica se la cartella static esiste
    if not os.path.exists(static_folder):
        print(f"ATTENZIONE: La cartella static non esiste in {static_folder}")

    # Percorsi delle immagini e loro configurazione
    icons_config = {
        "vegan": {"path": os.path.join(static_folder, "vegan.png"), "width": 24},
        "vegetarian": {"path": os.path.join(static_folder, "vegetarian_2.png"), "width": 24},
        "gluten_free": {"path": os.path.join(static_folder, "gluten_free_2.png"), "width": 24},
        "lactose_free": {"path": os.path.join(static_folder, "lactose_free_2.png"), "width": 24},
    }

    # Genera HTML per le icone
    # Questo blocco crea un dizionario di tag HTML per ogni icona
    # Se l'immagine non è disponibile, usa l'emoji di fallback
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
        """
        Formatta una riga di caratteristiche (icone) della ricetta.

        Args:
            features: Lista di stringhe HTML rappresentanti le caratteristiche

        Returns:
            str: HTML formattato per la riga di caratteristiche o stringa vuota se non ci sono caratteristiche
        """
        if not features:
            return ""
        row = '<div style="display: flex; align-items: center;">'
        row += "Caratteristiche: "
        row += ", ".join(features)
        row += '</div><br>'
        return row

    # Costruisci stringa preferenze per i messaggi
    # Questa sezione crea una rappresentazione delle preferenze dell'utente
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
        """
        Formatta la sezione degli ingredienti di una ricetta.

        Crea una lista HTML con tutti gli ingredienti e i loro contributi nutrizionali.

        Args:
            ingredients: Lista di oggetti CalculatedIngredient

        Returns:
            str: HTML formattato per la sezione ingredienti
        """
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
        """
        Formatta la sezione delle istruzioni di preparazione di una ricetta.

        Crea una lista ordinata HTML con tutti i passaggi di preparazione.

        Args:
            instructions: Lista di stringhe con i passaggi

        Returns:
            str: HTML formattato per la sezione istruzioni o stringa vuota se non ci sono istruzioni
        """
        if not instructions:
            return ""
        section = "<h3>Preparazione</h3><ol>"
        for step in instructions:
            section += f"<li>{step}</li>"
        section += "</ol>"
        return section

    # Funzione helper per creare le caratteristiche di una ricetta
    def get_recipe_features(recipe):
        """
        Ottiene le caratteristiche (icone) per una ricetta.

        Determina quali icone mostrare in base ai flag della ricetta, evitando
        ridondanze (es. non mostra "vegetariano" se la ricetta è già "vegana").

        Args:
            recipe: Oggetto FinalRecipeOption

        Returns:
            list: Lista di stringhe HTML rappresentanti le caratteristiche
        """
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
        """
        Formatta la sezione delle informazioni nutrizionali di una ricetta.
        Versione semplificata che mostra solo i CHO totali.

        Args:
            recipe: Oggetto FinalRecipeOption

        Returns:
            str: HTML formattato per la sezione nutrizione semplificata
        """
        section = "<h3>Informazioni nutrizionali</h3><ul>"
        section += f"<li><b>CHO Totali:</b> {recipe.total_cho:.1f}g</li>"
        # Rimuove la visualizzazione di calorie, proteine, grassi e fibre
        section += "</ul>"
        return section

    # Funzione helper per formattare una singola ricetta
    def format_recipe(recipe, index):
        """
        Formatta una singola ricetta completa.

        Combina tutte le sezioni (nome, descrizione, nutrizione, caratteristiche,
        ingredienti, istruzioni) in un'unica presentazione HTML.

        Args:
            recipe: Oggetto FinalRecipeOption
            index: Indice numerico della ricetta (per la numerazione)

        Returns:
            str: HTML formattato per l'intera ricetta
        """
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
    # Questa è la logica principale che determina quale tipo di output generare
    # in base al numero di ricette trovate e agli eventuali errori
    output_string = ""

    if final_recipes and len(final_recipes) >= min_recipes_required:
        # CASO 1: Successo - Abbiamo trovato abbastanza ricette
        output_string += f"<h1>Ricette personalizzate</h1><p>Ecco {len(final_recipes)} proposte di ricette che soddisfano i tuoi criteri (Target CHO: ~{preferences.target_cho:.1f}g, {prefs_string}):</p><hr>"

        for i, recipe in enumerate(final_recipes):
            output_string += format_recipe(recipe, i+1)

        output_string += "<h3>Suggerimenti</h3><p>Puoi modificare il target di carboidrati o le restrizioni dietetiche per ottenere ricette diverse.</p>"

    elif final_recipes:  # Trovate alcune ricette ma meno del minimo richiesto
        # CASO 2: Successo parziale - Abbiamo trovato alcune ricette ma non abbastanza
        output_string += f"<h1>Risultati parziali</h1><p>Spiacente, sono state trovate solo {len(final_recipes)} ricette che soddisfano tutti i tuoi criteri (Target CHO: ~{preferences.target_cho:.1f}g, {prefs_string}) invece delle {min_recipes_required} richieste:</p><hr>"

        for i, recipe in enumerate(final_recipes):
            output_string += format_recipe(recipe, i+1)

        output_string += "<h3>Suggerimenti</h3><p>Potresti provare a modificare leggermente il target di carboidrati o le restrizioni dietetiche per ottenere più opzioni.</p>"

    else:  # Nessuna ricetta trovata o errore grave precedente
        # CASO 3: Fallimento - Nessuna ricetta trovata
        output_string += f"<h1>Nessuna ricetta trovata</h1><p>Spiacente, non è stato possibile trovare nessuna ricetta che soddisfi tutti i tuoi criteri (Target CHO: ~{preferences.target_cho:.1f}g, {prefs_string}) entro la tolleranza richiesta.</p><hr>"

        if error_message:
            output_string += f"<p><b>Dettaglio problema:</b> {error_message}</p>"
        else:
            output_string += "<p><b>Possibile causa:</b> nessuna ricetta adatta è stata generata che rispetti tutti i vincoli specificati.</p>"
        # Potremmo rivedere i suggerimenti o  non mostrarli proprio
        output_string += "<h3>Suggerimenti</h3><p>Potresti provare a:</p><ul>"
        output_string += "<li>Modificare leggermente il target di carboidrati</li>"
        output_string += "<li>Rimuovere alcune delle restrizioni dietetiche, se possibile</li>"
        output_string += "<li>Verificare che i dataset degli ingredienti siano aggiornati e completi</li></ul>"
    # Aggiorna lo stato con l'output formattato
    new_state['final_output'] = output_string
    print(
        f"--- Formattazione completata ({len(output_string)} caratteri generati) ---")

    return new_state
