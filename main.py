# main.py (Aggiornato per il sistema semplificato "Generate then Fix")
import os
import argparse
import time
import pickle
from pprint import pprint
from dotenv import load_dotenv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, Optional, List

# Importa componenti dal progetto
from model_schema import UserPreferences, GraphState, FinalRecipeOption, IngredientInfo
from loaders import load_basic_ingredient_info
from workflow import create_workflow  # Usa il workflow aggiornato
from utils import normalize_name

# --- Configurazione ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
INGREDIENTS_FILE = os.path.join(DATA_DIR, "ingredients.csv")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "ingredients.index")
NAME_MAPPING_FILE = os.path.join(DATA_DIR, "ingredient_names.pkl")
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

# --- Funzione di Esecuzione Principale ---


def run_recipe_generation(
    initial_state: GraphState,
    streamlit_output: bool = False,
    img_dict: Optional[Dict[str, str]] = None,
) -> str:
    """
    Orchestra il processo di generazione ricette usando il workflow LangGraph.

    Args:
        initial_state: Stato iniziale già popolato
        streamlit_output: Flag per output formattato per Streamlit
        img_dict: Dizionario di immagini per output HTML

    Returns:
        Output formattato come stringa
    """
    print("\n--- Avvio Generazione Ricette (Generate then Fix) ---")
    start_time_workflow = time.time()

    # 1. Validazione Stato Iniziale
    required_keys = ['user_preferences', 'available_ingredients_data', 'embedding_model',
                     'normalize_function', 'faiss_index', 'index_to_name_mapping']
    missing_keys = [
        key for key in required_keys if key not in initial_state or initial_state.get(key) is None]
    if missing_keys:
        error_msg = f"Errore critico: Stato iniziale incompleto. Chiavi mancanti o None: {missing_keys}"
        print(error_msg)
        return error_msg

    # 2. Crea l'applicazione LangGraph
    print("--- Creazione Grafo Workflow ---")
    app = create_workflow()  # Usa il workflow aggiornato
    print("--- Grafo Creato ---")

    # 3. Esegui il Workflow passando lo stato iniziale
    print("--- Esecuzione Workflow LangGraph ---")
    try:
        final_state = app.invoke(initial_state)
        print("--- Esecuzione Workflow Completata ---")
    except Exception as wf_err:
        print(f"ERRORE DURANTE ESECUZIONE WORKFLOW: {wf_err}")
        import traceback
        traceback.print_exc()
        return f"Errore critico durante l'esecuzione del workflow: {wf_err}"

    # 4. Estrai e Restituisci l'Output Finale
    output_string = final_state.get(
        "final_output", "Nessun output formattato generato.")
    error_msg = final_state.get("error_message")

    # Aggiungi eventuale messaggio d'errore dal workflow all'output testuale
    if error_msg and not streamlit_output:
        output_string += f"\n\n*** Messaggio dal Workflow ***\n{error_msg}"

    end_time_workflow = time.time()
    print(
        f"--- Generazione e Workflow completati in {end_time_workflow - start_time_workflow:.2f} secondi ---")

    # Stampa riepilogo se non in modalità Streamlit
    if not streamlit_output:
        num_generated = len(final_state.get('generated_recipes', []))
        num_verified = len(final_state.get('final_verified_recipes', []))
        print(
            f"\nRiepilogo: Ricette generate={num_generated}, Ricette finali={num_verified}")
        if error_msg:
            print(f"Messaggio Workflow: {error_msg}\n")

    return output_string


# --- Esecuzione da Riga di Comando ---
if __name__ == "__main__":
    # Carica variabili d'ambiente (es. OPENAI_API_KEY)
    load_dotenv()

    # Configurazione Argomenti Command-Line
    parser = argparse.ArgumentParser(
        description="Generatore di Ricette con specifico contenuto CHO (Generate then Fix)")
    parser.add_argument("target_cho", type=float, help="Target CHO (g)")
    parser.add_argument("--vegan", action="store_true",
                        help="Solo ricette vegane")
    parser.add_argument("--vegetarian", action="store_true",
                        help="Solo ricette vegetariane")
    parser.add_argument("--gluten_free", action="store_true",
                        help="Solo ricette senza glutine")
    parser.add_argument("--lactose_free", action="store_true",
                        help="Solo ricette senza lattosio")
    args = parser.parse_args()

    # --- Caricamento Risorse per CLI ---
    print("--- Caricamento Risorse per CLI ---")
    try:
        print("Caricamento modello SBERT...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Caricamento indice FAISS...")
        faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        print("Caricamento mapping nomi...")
        with open(NAME_MAPPING_FILE, 'rb') as f:
            index_to_name_mapping = pickle.load(f)
        print("Caricamento info ingredienti...")
        available_ingredients_data = load_basic_ingredient_info(
            INGREDIENTS_FILE)

        # Verifica caricamenti
        if embedding_model is None or faiss_index is None or index_to_name_mapping is None or available_ingredients_data is None:
            raise RuntimeError(
                "Fallito caricamento di una o più risorse necessarie.")
        if faiss_index.ntotal != len(index_to_name_mapping):
            raise RuntimeError(
                f"Mismatch indice FAISS ({faiss_index.ntotal}) / mapping ({len(index_to_name_mapping)}).")

    except Exception as load_err:
        print(
            f"\nERRORE CRITICO durante caricamento risorse per CLI: {load_err}")
        print("Assicurati di aver eseguito 'python create_faiss_index.py' e che i file in 'data/' siano corretti.")
        exit(1)

    print("--- Risorse Caricate con Successo ---")

    # Crea oggetto UserPreferences
    prefs = UserPreferences(
        target_cho=args.target_cho,
        vegan=args.vegan,
        vegetarian=args.vegetarian or args.vegan,  # Vegano implica vegetariano
        gluten_free=args.gluten_free,
        lactose_free=args.lactose_free,
    )

    # --- Prepara Stato Iniziale per CLI ---
    cli_initial_state = GraphState(
        user_preferences=prefs,
        available_ingredients_data=available_ingredients_data,
        embedding_model=embedding_model,
        normalize_function=normalize_name,
        faiss_index=faiss_index,
        index_to_name_mapping=index_to_name_mapping,
        generated_recipes=[],
        final_verified_recipes=[],
        error_message=None,
        final_output=None
    )

    # --- Esecuzione Workflow da CLI ---
    print(f"\nAvvio generazione per CHO={args.target_cho}g, Prefs={prefs}")
    output = run_recipe_generation(
        initial_state=cli_initial_state,
        streamlit_output=False
    )
    print("\n--- Output Generato ---")
    print(output)
    print("--- Fine Esecuzione CLI ---")
