# main.py
import os
import argparse
import time
import pickle  # Per caricare mapping
from pprint import pprint
from dotenv import load_dotenv
import numpy as np
import faiss  # Importa FAISS
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, Optional, List

# Importa componenti dal progetto
from model_schema import UserPreferences, GraphState, FinalRecipeOption, IngredientInfo
from loaders import load_basic_ingredient_info  # Usa il nuovo loader base
from workflow import create_workflow
from utils import normalize_name  # Importa normalize

# --- Configurazione ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")  # Definisci anche qui se serve
INGREDIENTS_FILE = os.path.join(DATA_DIR, "ingredients.csv")
FAISS_INDEX_FILE = os.path.join(
    DATA_DIR, "ingredients.index")  # Percorso indice FAISS
NAME_MAPPING_FILE = os.path.join(
    DATA_DIR, "ingredient_names.pkl")  # Percorso mapping nomi
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'  # Modello SBERT

# --- Funzione di Esecuzione Principale (Firma Modificata) ---
# Ora accetta lo stato iniziale già popolato (da app.py o da main se eseguito da CLI)


def run_recipe_generation(
    initial_state: GraphState,  # Accetta lo stato pre-popolato
    streamlit_output: bool = False,
    img_dict: Optional[Dict[str, str]] = None,
) -> str:
    """
    Orchestra il processo di generazione ricette usando il workflow LangGraph.
    Ora riceve lo stato iniziale già pronto.
    """
    print("\n--- Avvio Generazione Ricette (FAISS) ---")
    start_time_workflow = time.time()

    # 1. Validazione Stato Iniziale (Opzionale ma utile)
    required_keys = ['user_preferences', 'available_ingredients_data', 'embedding_model',
                     'normalize_function', 'faiss_index', 'index_to_name_mapping']
    missing_keys = [
        key for key in required_keys if key not in initial_state or initial_state.get(key) is None]
    if missing_keys:
        error_msg = f"Errore critico: Stato iniziale incompleto. Chiavi mancanti o None: {missing_keys}"
        print(error_msg)
        return error_msg  # Esce se lo stato non è valido

    # 2. Crea l'applicazione LangGraph
    # Considera di creare il workflow una sola volta se la struttura è statica
    print("--- Creazione Grafo Workflow ---")
    app = create_workflow()
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


# --- Esecuzione da Riga di Comando (Modificata) ---
if __name__ == "__main__":
    # Carica variabili d'ambiente (es. OPENAI_API_KEY)
    load_dotenv()

    # Configurazione Argomenti Command-Line (come prima)
    parser = argparse.ArgumentParser(...)  # Come prima
    parser.add_argument("target_cho", type=float, help="Target CHO (g)")
    parser.add_argument("--vegan", action="store_true")
    parser.add_argument("--vegetarian", action="store_true")
    parser.add_argument("--gluten_free", action="store_true")
    parser.add_argument("--lactose_free", action="store_true")
    args = parser.parse_args()

    # --- Caricamento Risorse per CLI ---
    print("--- Caricamento Risorse per CLI (FAISS) ---")
    try:
        print("Caricamento modello SBERT...")
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME)  # Carica su CPU/MPS
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
        vegetarian=args.vegetarian or args.vegan,
        gluten_free=args.gluten_free,
        lactose_free=args.lactose_free,
    )

    # --- Prepara Stato Iniziale per CLI ---
    cli_initial_state = GraphState(
        user_preferences=prefs,
        available_ingredients_data=available_ingredients_data,
        embedding_model=embedding_model,
        normalize_function=normalize_name,  # Assumi normalize_name sia importato
        faiss_index=faiss_index,
        index_to_name_mapping=index_to_name_mapping,
        generated_recipes=[],
        final_verified_recipes=[],
        error_message=None,
        final_output=None
    )

    # --- Esecuzione Workflow da CLI ---
    print(
        f"\nAvvio generazione FAISS per CHO={args.target_cho}g, Prefs={prefs}")
    output = run_recipe_generation(
        initial_state=cli_initial_state,  # Passa lo stato
        streamlit_output=False
        # img_dict non serve per CLI
    )
    print("\n--- Output Generato ---")
    print(output)
    print("--- Fine Esecuzione CLI ---")
