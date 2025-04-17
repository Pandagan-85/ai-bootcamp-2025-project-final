"""
Script principale per l'esecuzione del sistema di generazione di ricette personalizzate.

Questo modulo contiene la funzione principale (run_recipe_generation) che orchestra 
l'intero processo di generazione delle ricette personalizzate. Il sistema prende le preferenze 
dell'utente (target di carboidrati e restrizioni dietetiche) e genera ricette 
che soddisfano quei criteri utilizzando un workflow basato su LangGraph.

Il modulo può essere eseguito direttamente dalla riga di comando o importato
e utilizzato dall'interfaccia web Streamlit (app.py).
"""
import os
import argparse  # Importa argparse
import time
from pprint import pprint
from dotenv import load_dotenv

#  Importa componenti dal progetto
from model_schema import UserPreferences, GraphState
from loaders import load_ingredients
from workflow import create_workflow

#  --- Configurazione ---
DATA_DIR = "data"
INGREDIENTS_FILE = os.path.join(DATA_DIR, "ingredients.csv")
RECIPES_FILE = os.path.join(DATA_DIR, "recipes.csv")


def run_recipe_generation(
    target_cho: float,
    vegan: bool,
    vegetarian: bool,
    gluten_free: bool,
    lactose_free: bool,
    max_recipes: int = None,
    streamlit_output: bool = False,  # Nuovo parametro
    streamlit_write=None,  # Nuovo parametro per Streamlit write
    streamlit_info=None,  # Nuovo parametro per Streamlit info
    streamlit_error=None,  # Nuovo parametro per Streamlit error
    img_dict=None,
) -> str:  # La funzione ora restituisce la stringa di output
    """
    Funzione principale che orchestra il processo di generazione delle ricette.

    Questa funzione esegue l'intero flusso di generazione di ricette personalizzate:
    1. Carica la chiave API OpenAI e verifica la configurazione
    2. Crea un oggetto preferenze utente con i parametri specificati
    3. Carica il database degli ingredienti disponibili
    4. Crea e compila il grafo di workflow LangGraph
    5. Esegue il workflow che genererà e verificherà le ricette
    6. Formatta e restituisce il risultato

    Args:
        target_cho: Target di carboidrati in grammi desiderato
        vegan: Flag per richiedere ricette vegane
        vegetarian: Flag per richiedere ricette vegetariane
        gluten_free: Flag per richiedere ricette senza glutine
        lactose_free: Flag per richiedere ricette senza lattosio
        max_recipes: Numero massimo di ricette da elaborare (opzionale)
        streamlit_output: Se True, formatta l'output per Streamlit
        streamlit_write: Funzione st.write di Streamlit (opzionale)
        streamlit_info: Funzione st.info di Streamlit (opzionale)
        streamlit_error: Funzione st.error di Streamlit (opzionale)
        img_dict: Dizionario con HTML per le icone (opzionale)

    Returns:
        L'output formattato (stringa HTML/Markdown)

    Note:
        - Questa funzione può essere utilizzata sia da interfaccia CLI che da interfaccia web
        - Il parametro streamlit_output modifica il comportamento per integrarsi con l'interfaccia web
        - Se utilizzata da riga di comando, stampa informazioni aggiuntive durante l'esecuzione
    """
    start_time = time.time()
    if not streamlit_output:
        print("--- Avvio Sistema Generazione Ricette ---")

    #  1. Carica Chiave API da .env
    if not streamlit_output:
        print("Caricamento variabili d'ambiente...")
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        error_message = "\nERRORE CRITICO: La variabile d'ambiente OPENAI_API_KEY non è impostata.\nAssicurati di avere un file .env nella directory principale con OPENAI_API_KEY='tua_chiave'"
        if streamlit_output and streamlit_error:
            streamlit_error(error_message)
        else:
            print(error_message)
        return ""  # Restituisce una stringa vuota per indicare errore

    #  2. Crea Oggetto Preferenze Utente
    user_preferences = UserPreferences(
        target_cho=target_cho,
        vegan=vegan,
        vegetarian=vegetarian,
        gluten_free=gluten_free,
        lactose_free=lactose_free,
    )
    if not streamlit_output:
        print("\nPreferenze Utente Impostate:")
        pprint(user_preferences.dict())

    #  3. Carica i Dataset
    if not streamlit_output:
        print(f"\nCaricamento ingredienti da: {INGREDIENTS_FILE}")
    ingredient_database = load_ingredients(INGREDIENTS_FILE)
    if not ingredient_database:
        error_message = f"ERRORE: Impossibile caricare il database degli ingredienti. Verifica il file {INGREDIENTS_FILE}."
        if streamlit_output and streamlit_error:
            streamlit_error(error_message)
        else:
            print(error_message)
        return ""

    #  4. Crea il Workflow (Grafo LangGraph)
    if not streamlit_output:
        print("\nCreazione e compilazione del workflow LangGraph...")
    app = create_workflow()

    #  5. Prepara lo Stato Iniziale per il Grafo
    initial_state = GraphState(
        user_preferences=user_preferences,
        available_ingredients=ingredient_database,
        initial_recipes=[],
        generated_recipes=[],
        final_verified_recipes=[],
        error_message=None,
        final_output=None,
        img_dict=img_dict,
    )
    if not streamlit_output:
        print("\nStato iniziale preparato per l'esecuzione del grafo.")

    #  6. Esegui il Workflow
    if not streamlit_output:
        print("\n--- ESECUZIONE WORKFLOW ---")
    final_state = app.invoke(initial_state)
    if not streamlit_output:
        print("--- WORKFLOW COMPLETATO ---")

    #  7. Mostra i Risultati
    if not streamlit_output:
        print("\n--- RISULTATO FINALE ---")

    output_string = ""  # Inizializza la stringa di output
    # Estrazione dell'output dal risultato del workflow
    if final_state and isinstance(final_state, dict):
        if 'final_output' in final_state and final_state['final_output']:
            # Utilizza l'output formattato generato dall'agente formatter
            output_string = final_state['final_output']
        else:
            # Fallback: formatta manualmente le ricette se l'agente formatter ha fallito
            output_string = "Non è stato possibile recuperare l'output formattato. Ecco le ricette trovate:\n"
            if 'final_verified_recipes' in final_state and final_state['final_verified_recipes']:
                recipes = final_state['final_verified_recipes']
                preferences = final_state['user_preferences']
                # Crea una stringa di preferenze per l'intestazione
                prefs_list = []
                if preferences.vegan:
                    prefs_list.append("Vegano")
                elif preferences.vegetarian:
                    prefs_list.append("Vegetariano")
                if preferences.gluten_free:
                    prefs_list.append("Senza Glutine")
                if preferences.lactose_free:
                    prefs_list.append("Senza Lattosio")
                prefs_string = ", ".join(
                    prefs_list) if prefs_list else "Nessuna preferenza specifica"

                output_string += f"\nEcco {len(recipes)} proposte di ricette che soddisfano i tuoi criteri (Target CHO: ~{preferences.target_cho:.1f}g, {prefs_string}):\n\n"
                # Formatta ogni ricetta in formato markdown
                for i, recipe in enumerate(recipes):
                    output_string += f"**{i+1}. {recipe.name}**\n"
                    output_string += f"* CHO Totali: {recipe.total_cho:.1f}g\n"
                    # Aggiungi flag dietetici
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
                    output_string += f"* Caratteristiche: {flags_string}\n\n"
                    # Aggiungi ingredienti
                    output_string += "* Ingredienti:\n"
                    for ing in recipe.ingredients:
                        output_string += f"    * {ing.name}: {ing.quantity_g:.1f}g (CHO: {ing.cho_contribution:.1f}g)\n"
                    output_string += "\n"
            else:
                output_string += "Nessuna ricetta verificata trovata.\n"
            # Aggiungi eventuale messaggio di errore
            if 'error_message' in final_state and final_state['error_message']:
                output_string += f"Errore: {final_state['error_message']}\n"
    else:
        output_string = "Errore: lo stato finale non è stato restituito correttamente.\n"

    #  Statistiche di esecuzione
    end_time = time.time()
    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    num_generated = len(final_state.get('generated_recipes', []))
    num_verified = len(final_state.get('final_verified_recipes', []))

    if not streamlit_output:
        print(f"\n--- Statistiche Esecuzione ---")
        print(f"Tempo totale: {minutes} minuti e {seconds} secondi")
        print(f"Ricette generate: {num_generated}")
        print(f"Ricette verificate e selezionate: {num_verified}")
        print("--- Esecuzione Terminata ---")
    else:
        if streamlit_info:
            streamlit_info(
                f"Tempo totale: {minutes} minuti e {seconds} secondi")
        if streamlit_write:
            streamlit_write(f"Ricette generate: {num_generated}")
            streamlit_write(
                f"Ricette verificate e selezionate: {num_verified}")

    return output_string


if __name__ == "__main__":
    #  Configurazione Argomenti Command-Line
    parser = argparse.ArgumentParser(
        description="Genera ricette basate su CHO e preferenze dietetiche.")
    parser.add_argument("target_cho", type=float,
                        help="Target di carboidrati in grammi (es. 80)")
    parser.add_argument("--vegan", action="store_true",
                        help="Filtra per ricette vegane")
    parser.add_argument("--vegetarian", action="store_true",
                        help="Filtra per ricette vegetariane")
    parser.add_argument("--gluten_free", action="store_true",
                        help="Filtra per ricette senza glutine")
    parser.add_argument("--lactose_free", action="store_true",
                        help="Filtra per ricette senza lattosio")
    parser.add_argument("--max_recipes", type=int, default=None,
                        help="Numero massimo di ricette da elaborare (opzionale)")

    args = parser.parse_args()

    #  Esecuzione da riga di comando
    output = run_recipe_generation(
        target_cho=args.target_cho,
        vegan=args.vegan,
        vegetarian=args.vegetarian or args.vegan,
        gluten_free=args.gluten_free,
        lactose_free=args.lactose_free,
        max_recipes=args.max_recipes,
        streamlit_output=False  # Indica che non è Streamlit
    )
    print(output)  # Stampa l'output sulla console
