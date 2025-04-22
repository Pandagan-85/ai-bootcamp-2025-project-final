from dotenv import load_dotenv
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
from PIL import Image
import pickle
import os
import time
import base64
import torch
torch.classes.__path__ = []  # per streamlit
load_dotenv()
# Importa funzioni e classi necessarie
try:
    from main import run_recipe_generation
    from model_schema import UserPreferences, GraphState
    from loaders import load_basic_ingredient_info, load_ingredient_database_with_mappings
    from utils import normalize_name
except ImportError as e:
    st.error(
        f"Errore import: {e}. Assicurati che tutti i file .py siano presenti e corretti.")
    st.stop()

# --- Definizione Costanti ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

INGREDIENTS_FILE = os.path.join(DATA_DIR, "ingredients.csv")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "ingredients.index")
NAME_MAPPING_FILE = os.path.join(DATA_DIR, "ingredient_names.pkl")
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

# Percorsi Immagini Statiche
VEGAN_IMG_PATH = os.path.join(STATIC_DIR, "vegan.png")
VEGETARIAN_IMG_PATH = os.path.join(STATIC_DIR, "vegetarian_2.png")
GLUTEN_FREE_IMG_PATH = os.path.join(STATIC_DIR, "gluten_free_2.png")
LACTOSE_FREE_IMG_PATH = os.path.join(STATIC_DIR, "lactose_free_2.png")
LOADING_IMG_PATH = os.path.join(STATIC_DIR, "loading.gif")

# Verifica Esistenza File Indice/Mapping
if not os.path.exists(FAISS_INDEX_FILE):
    st.error(f"Errore Critico: File indice FAISS non trovato: '{FAISS_INDEX_FILE}'. "
             f"Esegui lo script 'create_faiss_index.py' prima di avviare l'app.")
    st.stop()
if not os.path.exists(NAME_MAPPING_FILE):
    st.error(f"Errore Critico: File mapping nomi non trovato: '{NAME_MAPPING_FILE}'. "
             f"Esegui lo script 'create_faiss_index.py' prima di avviare l'app.")
    st.stop()

# --- Funzioni Helper ---


def get_base64_encoded_image(image_path: str) -> str | None:
    """
    Codifica un'immagine in base64 per incorporarla direttamente nell'HTML.

    Args:
        image_path: Percorso completo al file immagine

    Returns:
        Stringa dell'immagine codificata in base64 o None se l'immagine non esiste o si verifica un errore
    """
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def get_img_html(img_path: str, width: int = 24) -> str:
    """
    Codifica un'immagine in base64 per incorporarla direttamente nell'HTML.

    Args:
        image_path: Percorso completo al file immagine

    Returns:
        Stringa dell'immagine codificata in base64 o None se l'immagine non esiste o si verifica un errore
    """
    base64_image = get_base64_encoded_image(img_path)
    if base64_image:
        return f'<img src="data:image/png;base64,{base64_image}" width="{width}" style="margin-right: 5px; vertical-align: middle;" alt="{os.path.basename(img_path)}">'
    else:
        fallback_emojis = {"vegan.png": "🌱", "vegetarian.png": "🥗",
                           "gluten_free_2.png": "🌾", "lactose_free_2.png": "🥛"}
        emoji = fallback_emojis.get(os.path.basename(img_path), "⚠️")
        return f'<span title="Icona mancante: {os.path.basename(img_path)}" style="margin-right: 5px; vertical-align: middle;">{emoji}</span>'


def image_checkbox(label: str, img_path: str, img_width: int = 160, key: str | None = None, value: bool = False, text_below: bool = True) -> bool:
    """
    Crea un checkbox personalizzato con un'immagine sopra l'etichetta utilizzando componenti Streamlit.

    Args:
        label: Etichetta testuale del checkbox
        img_path: Percorso completo all'immagine da visualizzare
        img_width: Larghezza desiderata dell'immagine in pixel (default: 60)
        key: Chiave unica per il componente Streamlit (necessaria per state management)
        value: Valore iniziale del checkbox (default: False)
        text_below: Se True, posiziona il testo sotto l'immagine, altrimenti a fianco

    Returns:
        Stato corrente del checkbox (True se selezionato, False altrimenti)
    """
    container = st.container()
    img_col = container.container()
    img_html_for_display = ""
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path)
            img_col.image(img, width=img_width, use_container_width='auto')
        except Exception as e:
            print(f"Error displaying image {img_path}: {e}")
            fallback_emojis = {"vegan.png": "🌱", "vegetarian.png": "🥗",
                               "gluten_free_2.png": "🌾", "lactose_free_2.png": "🥛"}
            emoji = fallback_emojis.get(os.path.basename(img_path), "⚠️")
            img_col.markdown(
                f"<h1 style='text-align: center; margin-bottom: 5px;'>{emoji}</h1>", unsafe_allow_html=True)
    else:
        fallback_emojis = {"vegan.png": "🌱", "vegetarian.png": "🥗",
                           "gluten_free_2.png": "🌾", "lactose_free_2.png": "🥛"}
        emoji = fallback_emojis.get(os.path.basename(img_path), "⚠️")
        img_col.markdown(
            f"<h1 style='text-align: center; margin-bottom: 5px;'>{emoji}</h1>", unsafe_allow_html=True)
        if key == "first_run_check" and not st.session_state.get('missing_icon_warning_shown', False):
            st.warning(f"Icona non trovata: {img_path}. Verrà usata un'emoji.")
            st.session_state['missing_icon_warning_shown'] = True
    checkbox_col = container.container()
    checked = checkbox_col.checkbox(
        label, key=key, value=value, label_visibility="visible")
    return checked

# --- Funzioni di Caricamento Dati/Risorse con Cache ---


@st.cache_resource(show_spinner="Caricamento modello...")
def load_sbert_model_cached(model_name):
    print(f"--- Loading SentenceTransformer Model ({model_name}) ---")
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        st.error(f"Errore caricamento modello  '{model_name}': {e}.")
        return None


@st.cache_resource(show_spinner="Caricamento indice FAISS...")
def load_faiss_index_cached(index_path):
    print(f"--- Loading FAISS index from {index_path} ---")
    try:
        return faiss.read_index(index_path)
    except Exception as e:
        st.error(f"Errore caricamento indice FAISS da '{index_path}': {e}")
        return None


@st.cache_data(show_spinner="Caricamento mapping nomi...")
def load_name_mapping_cached(mapping_path):
    print(f"--- Loading name mapping from {mapping_path} ---")
    try:
        with open(mapping_path, 'rb') as f:
            name_mapping = pickle.load(f)
        if not isinstance(name_mapping, list):
            raise TypeError("Il file di mapping non contiene una lista.")
        print(f"Caricati {len(name_mapping)} nomi dal mapping.")
        return name_mapping
    except Exception as e:
        st.error(f"Errore caricamento mapping nomi da '{mapping_path}': {e}")
        return None


@st.cache_data(show_spinner="Caricamento info ingredienti...")
def load_basic_ingredient_info_cached(csv_filepath):
    print(f"--- Loading basic ingredient info from {csv_filepath} ---")
    data = load_basic_ingredient_info(csv_filepath)
    if data is None:
        st.error(f"Fallito caricamento dati ingredienti da {csv_filepath}.")
    return data


@st.cache_data(show_spinner="Caricamento info ingredienti con mappature...")
def load_ingredient_info_with_mappings_cached(csv_filepath):
    print(f"--- Loading ingredient info with mappings from {csv_filepath} ---")
    data, normalized_to_original, original_to_normalized = load_ingredient_database_with_mappings(
        csv_filepath)
    if data is None:
        st.error(f"Fallito caricamento dati ingredienti da {csv_filepath}.")
    return data, normalized_to_original, original_to_normalized


# --- Interfaccia Streamlit ---
st.set_page_config(
    page_title="NutriCHOice - Generatore Ricette CHO", layout="centered")
st.title("🥦 NutriCHOice la scelta intelligente, per un'alimentazione su misura) 🥕")

# Descrizione introduttiva con menzione Edgemony
st.markdown("""
### Come funziona NutriCHOice? 🔍

NutriCHOice è un assistente culinario intelligente che utilizza l'approccio innovativo **"Genera e Correggi"** per creare ricette personalizzate che rispettano i tuoi obiettivi nutrizionali.

**Sviluppato come progetto finale del Bootcamp AI di Edgemony** 🎓
""")

# Il processo in 4 fasi in markdown semplice per evitare sovrapposizioni
st.markdown("#### Il processo in 4 fasi:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    #### 🎯 Personalizzazione

    Imposti il tuo target di carboidrati e le preferenze alimentari (vegano, senza glutine, ecc.)
    """)

with col2:
    st.markdown("""
    #### 💡 Generazione Creativa

    Un agente AI genera ricette originali e gustose con creatività libera
    """)

with col3:
    st.markdown("""
    #### ✨ Verifica e Ottimizzazione

    FAISS confronta gli ingredienti, calcola i nutrienti e ottimizza le quantità
    """)

with col4:
    st.markdown("""
    #### 📑 Presentazione

    Le ricette validate vengono formattate con tutti i valori nutrizionali
    """)

st.markdown("")  # Spazio aggiuntivo

st.markdown("""
#### Perché scegliere NutriCHOice? 🌟

- **Precisione assoluta**: Valori nutrizionali basati su dati scientifici verificati
- **Creatività senza limiti**: Ricette originali e diverse ad ogni generazione
- **Rispetto delle preferenze**: Ogni ricetta è garantita conforme alle tue esigenze dietetiche
- **Ottimizzazione intelligente**: Le quantità vengono calibrate per raggiungere esattamente il tuo target di carboidrati
""")

st.markdown("---")
st.info("💡 Imposta le tue preferenze qui sotto e clicca sul pulsante per generare le tue ricette personalizzate!")
st.markdown("")


# --- Caricamento Risorse all'Avvio ---
start_load_time = time.time()
embedding_model = load_sbert_model_cached(EMBEDDING_MODEL_NAME)
faiss_index = load_faiss_index_cached(FAISS_INDEX_FILE)
index_to_name_mapping = load_name_mapping_cached(NAME_MAPPING_FILE)
available_ingredients_data, normalized_to_original, original_to_normalized = load_ingredient_info_with_mappings_cached(
    INGREDIENTS_FILE)
end_load_time = time.time()
print(
    f"Tempo caricamento risorse iniziali: {end_load_time - start_load_time:.2f} sec")

# Verifica Fallimento Caricamento Critico
if embedding_model is None or faiss_index is None or index_to_name_mapping is None or available_ingredients_data is None:
    st.error(
        "Errore critico durante il caricamento delle risorse necessarie. "
        "Controllare i log del terminale. L'applicazione non può continuare."
    )
    st.stop()

# Verifica consistenza indice/mapping
if faiss_index.ntotal != len(index_to_name_mapping):
    st.error(f"Errore Critico: Numero vettori nell'indice FAISS ({faiss_index.ntotal}) "
             f"non corrisponde alla lunghezza del mapping nomi ({len(index_to_name_mapping)}). "
             f"Rieseguire lo script 'create_faiss_index.py'.")
    st.stop()


st.header("Preferenze Ricetta")
target_cho = st.number_input(
    "🎯 Target Carboidrati (g/porzione) min 20gr, max 120gr",
    min_value=20.0,
    max_value=120.0,
    value=60.0,
    step=5.0,
    help="Imposta il contenuto desiderato di carboidrati per porzione (in grammi)."
)
# Validazione del valore CHO
cho_is_valid = 15.0 <= target_cho <= 130.0

if not cho_is_valid:
    st.error("⚠️ Il valore dei carboidrati deve essere compreso tra 15g e 130g.")

st.markdown("---")


# Flag per mostrare avviso solo una volta
if 'missing_icon_warning_shown' not in st.session_state:
    st.session_state['missing_icon_warning_shown'] = False

# Preferenze dietetiche nella pagina principale
st.subheader("Restrizioni Dietetiche")
st.write("Seleziona le tue preferenze:")
col1, col2, col3, col4 = st.columns(4)

# Checkbox personalizzati con immagini per le preferenze dietetiche
with col1:
    vegan = image_checkbox("Vegano", VEGAN_IMG_PATH,
                           key="vegan", text_below=True, value=False)
with col2:
    vegetarian = image_checkbox("Vegetariano", VEGETARIAN_IMG_PATH,
                                key="vegetarian", text_below=True, value=vegan)
with col3:
    gluten_free = image_checkbox("Senza Glutine", GLUTEN_FREE_IMG_PATH,
                                 key="gluten_free", text_below=True, value=False)
with col4:
    lactose_free = image_checkbox("Senza Lattosio", LACTOSE_FREE_IMG_PATH,
                                  key="lactose_free", text_below=True, value=False)

# Logica di aggiornamento: se vegan è selezionato, forza vegetarian a True
if vegan:
    vegetarian = True

# --- Pulsante di Generazione e Logica di Esecuzione ---
# Disabilita il pulsante se il valore CHO non è valido
button_disabled = not cho_is_valid

# Mostra un messaggio se il pulsante è disabilitato
if button_disabled:
    st.warning(
        "⚠️ Il pulsante 'Genera Ricette' è disabilitato perché il valore dei carboidrati è fuori range (15-130g).")

# Il pulsante è disabilitato se il valore è fuori range
if st.button("✨ Genera Ricette", use_container_width=True, type="primary", disabled=button_disabled):
    # Non serve più il doppio controllo perché il pulsante è già disabilitato

    st.markdown("---")
    results_container = st.container()
    placeholder = results_container.empty()

    # Mostra Indicatore Attesa
    loading_message = "🍳 Sto mescolando gli ingredienti... attendi un momento!"
    if os.path.exists(LOADING_IMG_PATH):
        try:
            with open(LOADING_IMG_PATH, "rb") as f:
                contents = f.read()
                data_url = base64.b64encode(contents).decode("utf-8")
            placeholder.markdown(
                f'<div style="text-align:center;"><img src="data:image/gif;base64,{data_url}" alt="loading..." width="200"><br>{loading_message}</div>', unsafe_allow_html=True)
        except Exception as e:
            placeholder.info(f"⏳ {loading_message}")
    else:
        placeholder.info(f"⏳ {loading_message}")

    start_time_workflow = time.time()

    # Prepara Dati per il Workflow
    user_preferences = UserPreferences(target_cho=target_cho, vegan=vegan,
                                       vegetarian=vegetarian, gluten_free=gluten_free, lactose_free=lactose_free)

    print(
        f"DEBUG APP - Preferenze selezionate: vegan={vegan}, vegetarian={vegetarian}, gluten_free={gluten_free}, lactose_free={lactose_free}")

    img_dict = {
        "vegan": get_img_html(VEGAN_IMG_PATH),
        "vegetarian": get_img_html(VEGETARIAN_IMG_PATH),
        "gluten_free": get_img_html(GLUTEN_FREE_IMG_PATH),
        "lactose_free": get_img_html(LACTOSE_FREE_IMG_PATH)
    }

    # --- Prepara Stato Iniziale per Workflow ---
    try:
        initial_state = GraphState(
            user_preferences=user_preferences,
            available_ingredients_data=available_ingredients_data,
            embedding_model=embedding_model,
            normalize_function=normalize_name,
            faiss_index=faiss_index,
            index_to_name_mapping=index_to_name_mapping,
            normalized_to_original=normalized_to_original,  # NUOVO
            original_to_normalized=original_to_normalized,  # NUOVO
            # Campi risultati inizializzati
            generated_recipes=[],
            final_verified_recipes=[],
            error_message=None,
            final_output=None
        )
    except Exception as state_err:
        st.error(
            f"Errore imprevisto nella creazione dello stato iniziale: {state_err}")
        st.stop()

    # --- ESEGUI WORKFLOW ---
    try:
        output_html = run_recipe_generation(
            initial_state=initial_state,
            streamlit_output=True,
            img_dict=img_dict
        )
        if output_html is None:
            output_html = "<p>Errore: Nessun output valido dal workflow.</p>"

    except Exception as workflow_error:
        st.error(
            f"Errore imprevisto durante l'esecuzione del workflow: {workflow_error}")
        st.exception(workflow_error)
        output_html = f"<div style='border: 2px solid red; padding: 10px; background-color: #ffeeee;'><h2>Errore Inaspettato</h2><pre>{workflow_error}</pre></div>"

    end_time_workflow = time.time()
    generation_time = end_time_workflow - start_time_workflow

    # --- Mostra Risultati ---
    placeholder.empty()
    results_container.success(
        f"🎉 Ricette elaborate in {generation_time:.2f} secondi!")
    results_container.markdown("---")
    results_container.markdown(output_html, unsafe_allow_html=True)

else:
    # Messaggio iniziale
    st.info("👋 Benvenuto! Imposta le tue preferenze e clicca '✨ Genera Ricette'.")

# --- Footer ---
st.markdown("---")
# Intestazione chiara per indicare che inizia il footer
st.markdown("### Chi siamo 🚀")

st.subheader("👥 Import Errror Domenico Not Found")
st.markdown("""
### I professionisti dietro NutriCHOice
*Un gruppo multidisciplinare dedicato a rivoluzionare la pianificazione alimentare personalizzata*
""")

st.markdown("<br>", unsafe_allow_html=True)

# 4 colonne per i 4 membri del team
team_col1, team_col2, team_col3, team_col4 = st.columns(4)

with team_col1:
    img_html_veronica = get_img_html("static/veronica.webp", width=100)
    st.markdown(f"""
    <div style='text-align: center;'>
        <div style='width:150px;height:150px;background-color:#f0f2f6;border-radius:50%;margin:auto;
                    overflow:hidden;display:flex;align-items:center;justify-content:center;'>
            {img_html_veronica}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Veronica Schembri")
    st.markdown("**AI & Machine Learning Lead**")
    st.markdown("Architetto dai mille talenti sviluppa la sua carriera tra design e programmazione. Divoratrice seriale di fumetti e di tutto lo scibile informatico e digitale.")
    st.markdown(
        "[LinkedIn](https://www.linkedin.com/in/veronicaschembri/) | [GitHub](https://github.com/Pandagan-85)")

with team_col2:
    img_html_francesca = get_img_html("static/francesca.webp", width=100)
    st.markdown(f"""
    <div style='text-align: center;'>
        <div style='width:150px;height:150px;background-color:#f0f2f6;border-radius:50%;margin:auto;
                    overflow:hidden;display:flex;align-items:center;justify-content:center;'>
            {img_html_francesca}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Francesca Ballirò")
    st.markdown("**AI & Machine Learning Lead**")
    st.markdown("Studentessa e imprenditrice, è abilissima nelle sfide lanciate dai dati. Riesce a manipolare complessi dataset anche sui mezzi pubblici.")
    st.markdown(
        "[LinkedIn](https://linkedin.com/placeholder) | [GitHub](https://github.com/placeholder)")

with team_col3:
    img_html_valentina = get_img_html("static/valentina.webp", width=100)
    st.markdown(f"""
    <div style='text-align: center;'>
        <div style='width:150px;height:150px;background-color:#f0f2f6;border-radius:50%;margin:auto;
                    overflow:hidden;display:flex;align-items:center;justify-content:center;'>
            {img_html_valentina}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Valentina Bellezza")
    st.markdown("**AI & Machine Learning Lead**")
    st.markdown("Specialista in marketing digitale, ossessionata dai KPI per ottimizzare le performance e massimizzare i ROAS, vive tra filosofia e salvifica praxis.")
    st.markdown(
        "[LinkedIn](https://linkedin.com/placeholder) | [GitHub](https://github.com/placeholder)")

with team_col4:

    img_html_giulia = get_img_html("static/lactose_free.png", width=100)
    st.markdown(f"""
    <div style='text-align: center;'>
        <div style='width:150px;height:150px;background-color:#f0f2f6;border-radius:50%;margin:auto;
                    overflow:hidden;display:flex;align-items:center;justify-content:center;'>
            {img_html_giulia}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Giulia Gambuzza")
    st.markdown("**AI & Machine Learning Lead**")
    st.markdown("Umanista digitale, incarna lo spirito del letterato di oggi, esperta nella manipolazione di corpora testuali, è anche dispensatrice compulsiva di materiale didattico.")
    st.markdown(
        "[LinkedIn](https://linkedin.com/placeholder) | [GitHub](https://github.com/placeholder)")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.caption(
    "© 2025 NutriCHOice v2.0 (Generate then Fix) - Tutti i diritti riservati")
