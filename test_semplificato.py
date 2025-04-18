# test_simplified.py
import os
import time
from dotenv import load_dotenv
import argparse

# Importa componenti dal progetto
from model_schema import UserPreferences, GraphState
from loaders import load_basic_ingredient_info  # Per caricare gli ingredienti
from workflow import create_workflow
from utils import normalize_name
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Configurazione
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INGREDIENTS_FILE = os.path.join(DATA_DIR, "ingredients.csv")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "ingredients.index")
NAME_MAPPING_FILE = os.path.join(DATA_DIR, "ingredient_names.pkl")
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'


def test_simplified_system(target_cho: float, vegan: bool = False, vegetarian: bool = False,
                           gluten_free: bool = False, lactose_free: bool = False):
    """
    Testa il sistema semplificato con l'approccio "Generate then Fix".

    Args:
        target_cho: Target CHO in grammi
        vegan, vegetarian, gluten_free, lactose_free: Preferenze dietetiche
    """
    print("\n" + "="*50)
    print(f"AVVIO TEST SISTEMA SEMPLIFICATO")
    print(f"Target CHO: {target_cho}g, Preferenze: " +
          f"vegan={vegan}, vegetarian={vegetarian}, gluten_free={gluten_free}, lactose_free={lactose_free}")
    print("="*50 + "\n")

    start_time = time.time()

    # Carica risorse necessarie
    print("--- Caricamento Risorse ---")
    try:
        # Carica modello di embedding
        print(f"Caricamento modello SBERT: {EMBEDDING_MODEL_NAME}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Carica indice FAISS
        print(f"Caricamento indice FAISS: {FAISS_INDEX_FILE}")
        if not os.path.exists(FAISS_INDEX_FILE):
            raise FileNotFoundError(f"File indice FAISS non trovato: {FAISS_INDEX_FILE}. " +
                                    "Esegui 'python create_faiss_index.py' prima di questo test.")
        faiss_index = faiss.read_index(FAISS_INDEX_FILE)

        # Carica mapping nomi
        print(f"Caricamento mapping nomi: {NAME_MAPPING_FILE}")
        if not os.path.exists(NAME_MAPPING_FILE):
            raise FileNotFoundError(f"File mapping nomi non trovato: {NAME_MAPPING_FILE}. " +
                                    "Esegui 'python create_faiss_index.py' prima di questo test.")
        with open(NAME_MAPPING_FILE, 'rb') as f:
            index_to_name_mapping = pickle.load(f)

        # Carica info ingredienti
        print(f"Caricamento info ingredienti: {INGREDIENTS_FILE}")
        if not os.path.exists(INGREDIENTS_FILE):
            raise FileNotFoundError(
                f"File ingredienti non trovato: {INGREDIENTS_FILE}")
        ingredient_data = load_basic_ingredient_info(INGREDIENTS_FILE)

        # Verifica consistenza indice/mapping
        if faiss_index.ntotal != len(index_to_name_mapping):
            raise ValueError(f"Mismatch tra dimensione indice FAISS ({faiss_index.ntotal}) " +
                             f"e mapping nomi ({len(index_to_name_mapping)})")

    except Exception as e:
        print(f"ERRORE DURANTE CARICAMENTO RISORSE: {e}")
        print("Impossibile procedere con il test.")
        return

    resources_time = time.time()
    print(f"Risorse caricate in {resources_time - start_time:.2f} secondi")

    # Crea preferenze
    prefs = UserPreferences(
        target_cho=target_cho,
        vegan=vegan,
        vegetarian=vegetarian or vegan,  # Vegano implica vegetariano
        gluten_free=gluten_free,
        lactose_free=lactose_free
    )

    # Prepara stato iniziale
    initial_state = GraphState(
        user_preferences=prefs,
        available_ingredients_data=ingredient_data,
        embedding_model=embedding_model,
        normalize_function=normalize_name,
        faiss_index=faiss_index,
        index_to_name_mapping=index_to_name_mapping,
        generated_recipes=[],
        final_verified_recipes=[],
        error_message=None,
        final_output=None
    )

    # Crea e esegui workflow
    print("\n--- Creazione Workflow ---")
    app = create_workflow()

    print("\n--- Esecuzione Workflow ---")
    workflow_start = time.time()
    try:
        final_state = app.invoke(initial_state)
        workflow_end = time.time()
        print(
            f"Workflow eseguito in {workflow_end - workflow_start:.2f} secondi")

        # Analizza risultati
        num_generated = len(final_state.get('generated_recipes', []))
        num_verified = len(final_state.get('final_verified_recipes', []))
        error_msg = final_state.get('error_message')

        print("\n" + "="*50)
        print("RISULTATI TEST")
        print(f"Ricette generate: {num_generated}")
        print(f"Ricette verificate e ottimizzate: {num_verified}")
        if error_msg:
            print(f"Messaggio di errore: {error_msg}")

        # Mostra dettagli ricette verificate
        if num_verified > 0:
            print("\nRICETTE VERIFICATE:")
            for i, recipe in enumerate(final_state.get('final_verified_recipes', [])):
                print(f"\n{i+1}. {recipe.name}")
                print(
                    f"   CHO: {recipe.total_cho:.1f}g (Target: {target_cho}g)")
                print(f"   Ingredienti: {len(recipe.ingredients)}")
                print(f"   Proprietà: vegan={recipe.is_vegan}, vegetarian={recipe.is_vegetarian}, " +
                      f"gluten_free={recipe.is_gluten_free}, lactose_free={recipe.is_lactose_free}")

        # Mostra output formattato (opzionale)
        output = final_state.get('final_output')
        if output:
            # Mostra solo l'inizio per brevità
            print("\nINIZIO OUTPUT FORMATTATO:")
            print(output[:500] + "...")

        print("\nTEST COMPLETATO")
        total_time = time.time() - start_time
        print(f"Tempo totale: {total_time:.2f} secondi")
        print("="*50)

    except Exception as e:
        print(f"ERRORE DURANTE ESECUZIONE WORKFLOW: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Carica variabili d'ambiente (es. OPENAI_API_KEY)
    load_dotenv()

    # Controlla che OPENAI_API_KEY sia impostata
    if not os.getenv("OPENAI_API_KEY"):
        print("ERRORE: OPENAI_API_KEY non impostata. Imposta la variabile d'ambiente prima di eseguire questo test.")
        exit(1)

    # Parsing argomenti command-line
    parser = argparse.ArgumentParser(
        description="Test del sistema semplificato Generate-then-Fix")
    parser.add_argument("target_cho", type=float, help="Target CHO (g)")
    parser.add_argument("--vegan", action="store_true",
                        help="Preferenza vegana")
    parser.add_argument("--vegetarian", action="store_true",
                        help="Preferenza vegetariana")
    parser.add_argument("--gluten_free", action="store_true",
                        help="Preferenza senza glutine")
    parser.add_argument("--lactose_free", action="store_true",
                        help="Preferenza senza lattosio")

    args = parser.parse_args()

    # Esegui test
    test_simplified_system(
        target_cho=args.target_cho,
        vegan=args.vegan,
        vegetarian=args.vegetarian,
        gluten_free=args.gluten_free,
        lactose_free=args.lactose_free
    )
