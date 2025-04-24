import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os  # Aggiunto import os
import base64
from PIL import Image  # Aggiunto import Image
from loaders import load_basic_ingredient_info, load_ingredient_database_with_mappings

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
def load_sbert_model_cached(model_name: str) -> SentenceTransformer | None:
    """
    Carica e memorizza in cache il modello SentenceTransformer.

    Args:
        model_name: Nome del modello SentenceTransformer da caricare.

    Returns:
        Il modello SentenceTransformer caricato, o None se si verifica un errore.
    """
    print(f"--- Loading SentenceTransformer Model ({model_name}) ---")
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        st.error(f"Errore caricamento modello  '{model_name}': {e}.")
        return None


@st.cache_resource(show_spinner="Caricamento indice FAISS...")
def load_faiss_index_cached(index_path: str) -> faiss.Index | None:
    """
    Carica e memorizza in cache l'indice FAISS.

    Args:
        index_path: Percorso al file dell'indice FAISS.

    Returns:
        L'indice FAISS caricato, o None se si verifica un errore.
    """
    print(f"--- Loading FAISS index from {index_path} ---")
    try:
        return faiss.read_index(index_path)
    except Exception as e:
        st.error(f"Errore caricamento indice FAISS da '{index_path}': {e}")
        return None


@st.cache_data(show_spinner="Caricamento mapping nomi...")
def load_name_mapping_cached(mapping_path: str) -> list[str] | None:
    """
    Carica e memorizza in cache il mapping dei nomi degli ingredienti.

    Args:
        mapping_path: Percorso al file pickle contenente il mapping dei nomi.

    Returns:
        La lista del mapping dei nomi caricata, o None se si verifica un errore.
    """
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
def load_basic_ingredient_info_cached(csv_filepath: str) -> dict[str, dict] | None:
    """
    Carica e memorizza in cache le informazioni di base sugli ingredienti.

    Args:
        csv_filepath: Percorso al file CSV contenente le informazioni sugli ingredienti.

    Returns:
        Un dizionario contenente le informazioni sugli ingredienti,
        o None se si verifica un errore.
    """
    print(f"--- Loading basic ingredient info from {csv_filepath} ---")
    data = load_basic_ingredient_info(csv_filepath)
    if data is None:
        st.error(f"Fallito caricamento dati ingredienti da {csv_filepath}.")
    return data


@st.cache_data(show_spinner="Caricamento info ingredienti con mappature...")
def load_ingredient_info_with_mappings_cached(csv_filepath: str) -> tuple[dict[str, dict], dict[str, str], dict[str, str]] | None:
    """
    Carica e memorizza in cache le informazioni sugli ingredienti con mappature.

    Args:
        csv_filepath: Percorso al file CSV contenente le informazioni sugli ingredienti.

    Returns:
        Una tupla contenente:
        - Un dizionario con le informazioni sugli ingredienti.
        - Un dizionario per la mappatura dei nomi normalizzati ai nomi originali.
        - Un dizionario per la mappatura dei nomi originali ai nomi normalizzati,
        o None se si verifica un errore.
    """
    print(f"--- Loading ingredient info with mappings from {csv_filepath} ---")
    data, normalized_to_original, original_to_normalized = load_ingredient_database_with_mappings(
        csv_filepath)
    if data is None:
        st.error(f"Fallito caricamento dati ingredienti da {csv_filepath}.")
    return data, normalized_to_original, original_to_normalized
