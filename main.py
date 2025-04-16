# main.py

import os
import argparse
import time
from pprint import pprint
from dotenv import load_dotenv

# Importa componenti dal progetto
from model_schema import UserPreferences, GraphState
from loaders import load_ingredients, load_recipes
from workflow import create_workflow

# --- Configurazione ---
DATA_DIR = "data"
INGREDIENTS_FILE = os.path.join(DATA_DIR, "ingredients.csv")
RECIPES_FILE = os.path.join(DATA_DIR, "recipes.csv")


def run_recipe_generation(target_cho: float, vegan: bool, vegetarian: bool, gluten_free: bool, lactose_free: bool, max_recipes: int = None):
    """
    Funzione principale che orchestra il processo di generazione delle ricette.

    Args:
        target_cho: Target di carboidrati in grammi
        vegan: Flag per ricette vegane
        vegetarian: Flag per ricette vegetariane
        gluten_free: Flag per ricette senza glutine
        lactose_free: Flag per ricette senza lattosio
        max_recipes: Numero massimo di ricette da elaborare (per limitare il tempo di esecuzione)
    """
    start_time = time.time()
    print("--- Avvio Sistema Generazione Ricette ---")

    # 1. Carica Chiave API da .env
    print("Caricamento variabili d'ambiente...")
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("\nERRORE CRITICO: La variabile d'ambiente OPENAI_API_KEY non è impostata.")
        print("Assicurati di avere un file .env nella directory principale con OPENAI_API_KEY='tua_chiave'")
        return  # Interrompe l'esecuzione

    # 2. Crea Oggetto Preferenze Utente
    user_preferences = UserPreferences(
        target_cho=target_cho,
        vegan=vegan,
        # Nota: se vegan è True, vegetarian dovrebbe esserlo implicitamente, ma lo teniamo separato come da input
        vegetarian=vegetarian,
        gluten_free=gluten_free,
        lactose_free=lactose_free
    )
    print("\nPreferenze Utente Impostate:")
    pprint(user_preferences.dict())

    # 3. Carica i Dataset
    print(f"\nCaricamento ingredienti da: {INGREDIENTS_FILE}")
    ingredient_database = load_ingredients(INGREDIENTS_FILE)
    if not ingredient_database:
        print(
            f"ERRORE: Impossibile caricare il database degli ingredienti. Verifica il file {INGREDIENTS_FILE}.")
        return

    # 4. Crea il Workflow (Grafo LangGraph)
    print("\nCreazione e compilazione del workflow LangGraph...")
    app = create_workflow()  # La funzione create_workflow compila già il grafo

    # 5. Prepara lo Stato Iniziale per il Grafo
    # Deve contenere tutte le chiavi definite in GraphState
    initial_state = GraphState(
        user_preferences=user_preferences,
        available_ingredients=ingredient_database,
        initial_recipes=[],            # Non utilizziamo più ricette iniziali
        generated_recipes=[],          # Inizialmente vuoto
        final_verified_recipes=[],     # Inizialmente vuoto
        error_message=None,            # Nessun errore all'inizio
        final_output=None              # Nessun output all'inizio
    )
    print("\nStato iniziale preparato per l'esecuzione del grafo.")
    # pprint(initial_state) # Decommenta per debug dello stato iniziale

    # 6. Esegui il Workflow
    print("\n--- ESECUZIONE WORKFLOW ---")
    # L'esecuzione passa lo stato iniziale attraverso i nodi e le decisioni
    # fino a raggiungere lo stato finale (END)
    final_state = app.invoke(initial_state)
    print("--- WORKFLOW COMPLETATO ---")

    # 7. Mostra i Risultati
    print("\n--- RISULTATO FINALE ---")
    if final_state and isinstance(final_state, dict):
        if 'final_output' in final_state and final_state['final_output']:
            print(final_state['final_output'])
        else:
            # Genera un output alternativo se l'output non è stato salvato correttamente
            print(
                "Non è stato possibile recuperare l'output formattato. Ecco le ricette trovate:")
            # Stampa le ricette verificate direttamente
            if 'final_verified_recipes' in final_state and final_state['final_verified_recipes']:
                recipes = final_state['final_verified_recipes']
                preferences = final_state['user_preferences']

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

                print(
                    f"\nEcco {len(recipes)} proposte di ricette che soddisfano i tuoi criteri (Target CHO: ~{preferences.target_cho:.1f}g, {prefs_string}):\n")

                for i, recipe in enumerate(recipes):
                    print(f"**{i+1}. {recipe.name}**")
                    print(f"* CHO Totali: {recipe.total_cho:.1f}g")

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
                    print(f"* Caratteristiche: {flags_string}")

                    print("* Ingredienti:")
                    for ing in recipe.ingredients:
                        print(
                            f"    * {ing.name}: {ing.quantity_g:.1f}g (CHO: {ing.cho_contribution:.1f}g)")
                    print()
            else:
                print("Nessuna ricetta verificata trovata.")

            # Stampa il messaggio di errore se presente
            if 'error_message' in final_state and final_state['error_message']:
                print(f"Errore: {final_state['error_message']}")
    else:
        print("Errore: lo stato finale non è stato restituito correttamente.")

    # Statistiche di esecuzione
    end_time = time.time()
    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    # Statistiche sulla quantità di ricette
    num_generated = len(final_state.get('generated_recipes', []))
    num_verified = len(final_state.get('final_verified_recipes', []))

    print(f"\n--- Statistiche Esecuzione ---")
    print(f"Tempo totale: {minutes} minuti e {seconds} secondi")
    print(f"Ricette generate: {num_generated}")
    print(f"Ricette verificate e selezionate: {num_verified}")
    print("--- Esecuzione Terminata ---")


# --- Punto di Ingresso dello Script ---
if __name__ == "__main__":
    # Configurazione Argomenti Command-Line (Alternativa all'hardcoding)
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
                        help="Numero massimo di ricette da elaborare (opzionale, per limitare il tempo di esecuzione)")

    args = parser.parse_args()

    # Esempio di chiamata diretta (se non si usano argomenti da linea di comando)
    # run_recipe_generation(target_cho=80, vegan=True, vegetarian=True, gluten_free=False, lactose_free=True)

    # Chiamata con argomenti da linea di comando
    run_recipe_generation(
        target_cho=args.target_cho,
        vegan=args.vegan,
        # Una ricetta vegana è anche vegetariana
        vegetarian=args.vegetarian or args.vegan,
        gluten_free=args.gluten_free,
        lactose_free=args.lactose_free,
        max_recipes=args.max_recipes
    )
