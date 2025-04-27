"""
app.py - Interfaccia Streamlit per NutriCHOice

Questo modulo gestisce l'interfaccia utente dell'applicazione NutriCHOice,
permettendo all'utente di selezionare le proprie preferenze nutrizionali,
generare ricette personalizzate con un agente AI, e scaricare il risultato finale.

Componenti principali:
- Caricamento risorse (modello SBERT, FAISS, ingredienti)
- UI per input preferenze (CHO, vegan, vegetariano, ecc.)
- Workflow AI: Generazione → Verifica → Formattazione
- Visualizzazione output HTML e pulsante download

Autori: Team NutriCHOice - Progetto finale Edgemony AI Bootcamp 2025
"""

# === IMPORT DI LIBRERIE E CONFIGURAZIONI INIZIALI ===
from PIL import Image  # Aggiunto import Image
import base64
import os
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, Any, List
import numpy as np
import time
import torch


torch.classes.__path__ = []  # per streamlit
load_dotenv()


# Importa funzioni e classi necessarie
try:
    from main import run_recipe_generation
    from model_schema import UserPreferences, GraphState
    from utils import normalize_name
    from utils_app import get_base64_encoded_image, get_img_html, image_checkbox, \
        load_sbert_model_cached, load_faiss_index_cached, load_name_mapping_cached, \
        load_basic_ingredient_info_cached, load_ingredient_info_with_mappings_cached
    from download_recipes import add_download_button
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


# --- Interfaccia Streamlit ---
st.set_page_config(
    page_title="NutriCHOice - Generatore Ricette CHO", layout="centered")
# --- Header con logo e titoli ---
col_logo, col_title = st.columns([1, 4])

with col_logo:
    st.image('static/logo.webp',  width=200)

with col_title:
    st.image('static/NutriCHOice.webp', width=300)
    st.markdown("""
    #### _La scelta intelligente per un'alimentazione su misura_
    """)


# Descrizione introduttiva con menzione Edgemony
st.markdown("""
### Come funziona NutriCHOice? 🔍

NutriCHOice è un assistente culinario intelligente che utilizza l'approccio innovativo **"Genera e Correggi"** per creare ricette personalizzate che rispettano i tuoi obiettivi nutrizionali.

**Sviluppato come progetto finale del Bootcamp AI di Edgemony** 🎓
""")

# Il processo in 4 fasi in markdown semplice per evitare sovrapposizioni
st.markdown('---')
st.markdown("#### Il processo in 4 fasi:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <p style='font-size: 36px; margin-bottom:-20px'>🎯</p>
        <h4 style='font-size: 16px'>Personalizzazione</h4>
        <p>
            Imposti il tuo target di carboidrati e le preferenze alimentari (vegano, senza glutine, ecc.)
        </p>
        """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <p style='font-size: 36px; margin-bottom:-20px'>💡</p> 
    <h4 style='font-size: 16px'>Generazione Creativa</h4>

    <p>Un agente AI genera ricette originali e gustose con creatività libera</p>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <p style='font-size: 36px; margin-bottom:-20px'>✨</p>
    <h4 style='font-size: 16px'>Verifica e Ottimizzazione</h4>

    <p>FAISS confronta gli ingredienti, calcola i nutrienti e ottimizza le quantità</p>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <p style='font-size: 36px; margin-bottom:-20px'>📑<p>
    <h4 style='font-size: 16px'>Presentazione</h4>

    <p>Le ricette validate vengono formattate con tutti i valori nutrizionali</p>
    """, unsafe_allow_html=True)
st.markdown('---')
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
start_load_time = time.time()  # Definisci start_load_time qui
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

# Inizializza le variabili di sessione
if 'output_html' not in st.session_state:
    st.session_state.output_html = None
if 'has_recipes' not in st.session_state:
    st.session_state.has_recipes = False
if 'clear_recipes' not in st.session_state:
    st.session_state.clear_recipes = False
if 'missing_icon_warning_shown' not in st.session_state:
    st.session_state['missing_icon_warning_shown'] = False

# Crea una key per mantenere le ricette visibili
if 'display_results' not in st.session_state:
    st.session_state.display_results = False

st.header("Preferenze Ricetta")
st.subheader("🎯 Target Carboidrati (CHO in grammi)")
st.markdown("min 20gr, max 140gr")
target_cho = st.number_input(
    label="",
    min_value=20.0,
    max_value=140.0,
    value=60.0,
    step=5.0,
    help="Imposta il contenuto desiderato di carboidrati per porzione (in grammi)."
)
# Validazione del valore CHO
cho_is_valid = 15.0 <= target_cho <= 140.0

if not cho_is_valid:
    st.error("⚠️ Il valore dei carboidrati deve essere compreso tra 15g e 130g.")

st.markdown("---")

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
generate_button = st.button(
    "✨ Genera Ricette", use_container_width=True, type="primary", disabled=button_disabled)

# Creiamo un container per i risultati dopo il pulsante
results_container = st.container()

# Logica per processare la generazione delle ricette
if generate_button:
    # Se si preme il pulsante, dobbiamo generare nuove ricette
    st.session_state.clear_recipes = True
    st.session_state.display_results = True

    # Placeholder per il messaggio di caricamento
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
        f"DEBUG APP - Preferenze selezionate: target_cho={target_cho} vegan={vegan}, vegetarian={vegetarian}, gluten_free={gluten_free}, lactose_free={lactose_free}")

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
        else:
            # Memorizza l'HTML generato nella sessione per utilizzo nel download
            st.session_state.output_html = output_html
            # Verifica se ci sono ricette nell'output
            st.session_state.has_recipes = "Ricette personalizzate" in output_html or "Risultati parziali" in output_html

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

    # Aggiungi pulsante di download solo se ci sono ricette
    if st.session_state.has_recipes:
        download_container = results_container.container()
        download_container.markdown("---")
        download_container.info(
            "📥 Vuoi salvare queste ricette sul tuo dispositivo?")
        col1, col2 = download_container.columns([1, 3])
        with col1:
            # Aggiungi pulsante download
            add_download_button(output_html, container=col1)
        with col2:
            col2.caption(
                "Le ricette verranno salvate in formato testo per consultazione offline.")

# Mostra le ricette esistenti se sono state generate in precedenza e non si è premuto "Genera Ricette"
elif st.session_state.has_recipes and st.session_state.output_html is not None:
    if st.session_state.display_results:
        # Mostra le ricette precedentemente generate
        results_container.markdown("---")
        results_container.info(
            "📋 Ecco le tue ricette precedentemente generate")
        results_container.markdown(
            st.session_state.output_html, unsafe_allow_html=True)

        # Mostra pulsante download
        download_container = results_container.container()
        download_container.markdown("---")
        download_container.info(
            "📥 Vuoi salvare queste ricette sul tuo dispositivo?")
        col1, col2 = download_container.columns([1, 3])
        with col1:
            add_download_button(st.session_state.output_html, container=col1)
        with col2:
            col2.caption(
                "Le ricette verranno salvate in formato testo per consultazione offline.")
else:
    # Messaggio iniziale
    st.info("👋 Benvenuto! Imposta le tue preferenze e clicca '✨ Genera Ricette'.")

# --- Footer ---
st.markdown("---")
# Intestazione chiara per indicare che inizia il footer
st.markdown("### Chi siamo 🚀")

st.subheader("👥 Import Error Domenico Not Found")
st.markdown("""
### I professionisti dietro NutriCHOice
*Un gruppo multidisciplinare dedicato a rivoluzionare la pianificazione alimentare personalizzata*
""")

st.markdown("<br>", unsafe_allow_html=True)

# 4 colonne per i 4 membri del team
team_col1, team_col2, team_col3, team_col4 = st.columns(4)

with team_col1:
    img_html_veronica = get_img_html(
        "static/Veronica.webp", width=100)
    st.markdown(f"""
    <div style='text-align: center;'>
        <div style='width:150px;height:150px;background-color:#f0f2f6;border-radius:50%;margin:auto;
                    overflow:hidden;display:flex;align-items:center;justify-content:center;'>
            {img_html_veronica}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Veronica Schembri")
    st.markdown("Architetto dai mille talenti sviluppa la sua carriera tra design e programmazione. Divoratrice seriale di fumetti e di tutto lo scibile informatico e digitale.")
    st.markdown(
        "[LinkedIn](https://www.linkedin.com/in/veronicaschembri/) | [GitHub](https://github.com/Pandagan-85)")

with team_col2:
    img_html_francesca = get_img_html(
        "static/Francesca.webp", width=100)
    st.markdown(f"""
    <div style='text-align: center;'>
        <div style='width:150px;height:150px;background-color:#f0f2f6;border-radius:50%;margin:auto;
                    overflow:hidden;display:flex;align-items:center;justify-content:center;'>
            {img_html_francesca}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Francesca Ballirò")
    st.markdown("Studentessa e imprenditrice, è felicissima quando rileva bug durante i test ma è altrettanto abile nel trovare soluzioni ottimali.")
    st.markdown(
        "[LinkedIn](https://www.linkedin.com/in/francesca-ballir%C3%B2-060b92331/) | [GitHub](https://github.com/francescaballiro)")

with team_col3:
    img_html_valentina = get_img_html(
        "static/valentina.webp", width=100)
    st.markdown(f"""
    <div style='text-align: center;'>
        <div style='width:150px;height:150px;background-color:#f0f2f6;border-radius:50%;margin:auto;
                    overflow:hidden;display:flex;align-items:center;justify-content:center;'>
            {img_html_valentina}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Valentina Bellezza")
    st.markdown("Specialista in marketing digitale, ossessionata dai KPI per ottimizzare le performance e massimizzare i ROAS, vive tra filosofia e salvifica praxis.")
    st.markdown(
        "[LinkedIn](https://www.linkedin.com/in/valentinabellezza/) | [GitHub](https://github.com/Valentina-create)")

with team_col4:

    img_html_giulia = get_img_html("static/giulia.webp", width=100)
    st.markdown(f"""
    <div style='text-align: center;'>
        <div style='width:150px;height:150px;background-color:#f0f2f6;border-radius:50%;margin:auto;
                    overflow:hidden;display:flex;align-items:center;justify-content:center;'>
            {img_html_giulia}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Giulia Gambuzza")
    st.markdown("Umanista digitale, incarna lo spirito del letterato di oggi, esperta nella manipolazione di corpora testuali, è anche dispensatrice compulsiva di materiale didattico.")
    st.markdown(
        "[LinkedIn](https://www.linkedin.com/in/giulia-g-71b19613b/) | [GitHub](https://github.com/KaguyaHime905)")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.caption(
    "© 2025 NutriCHOice v2.0 (Generate then Fix) - Tutti i diritti riservati")
