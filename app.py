"""
Applicazione web Streamlit per il sistema di generazione ricette.

Questo modulo implementa l'interfaccia web usando Streamlit per il sistema di generazione ricette.
Consente agli utenti di inserire le loro preferenze (target CHO e restrizioni dietetiche)
attraverso un'interfaccia grafica user-friendly e mostra le ricette generate in HTML.

L'interfaccia include:
- Selezione del target di carboidrati (CHO)
- Opzioni per restrizioni dietetiche (vegano, vegetariano, senza glutine, senza lattosio)
- Visualizzazione delle ricette generate con formattazione HTML
- Indicatori di avanzamento durante la generazione
"""
import time
import os
import base64
from PIL import Image

import streamlit as st
from main import run_recipe_generation


# Create a proper path to access static files in Streamlit


def get_base64_encoded_image(image_path):
    """
    Converte un'immagine in una stringa base64 per l'embedding in HTML.

    Args:
        image_path: Percorso all'immagine da codificare

    Returns:
        str: Stringa base64 dell'immagine

    Note:
        - Utilizzata per incorporare icone direttamente nell'HTML dell'output
        - La codifica base64 permette di visualizzare immagini senza salvarle 
          come file separati, ideale per interfacce web dinamiche
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


# Impostazione del percorso alla cartella static
# È necessario creare una cartella 'static' con le immagini delle icone all'interno
# La cartella static deve essere nella stessa directory dell'app Streamlit
static_folder = "static"

# Assicurati che la cartella static esista
if not os.path.exists(static_folder):
    os.makedirs(static_folder)
    st.warning(
        f"Created '{static_folder}' folder. Please place your icon images in this folder.")

# Percorsi delle immagini (relativi alla cartella static)
vegan_img_path = os.path.join(static_folder, "vegan.png")
vegetarian_img_path = os.path.join(static_folder, "vegetarian.png")
gluten_free_img_path = os.path.join(static_folder, "gluten_free.png")
lactose_free_img_path = os.path.join(static_folder, "lactose_free.png")
loading_img_path = os.path.join(static_folder, "loading.gif")

# Follia per creare i checkbok con img custom


def image_checkbox(label, img_path, img_width=150, key=None, value=False, text_below=True):
    """
    Crea un checkbox personalizzato con un'immagine.

    Questa funzione crea un elemento UI personalizzato che combina un'immagine
    e un checkbox, per un'interfaccia più intuitiva per le opzioni dietetiche.

    Args:
        label (str): Il testo da visualizzare.
        img_path (str): Il percorso all'immagine.
        img_width (int, optional): La larghezza desiderata per l'immagine. Default 150.
        key (str, optional): Una chiave univoca per il widget. Default None.
        value (bool, optional): Il valore iniziale del checkbox. Default False.
        text_below (bool, optional): Se True, mostra il testo sotto l'immagine. Default True.

    Returns:
        bool: True se il checkbox è selezionato, False altrimenti.

    Note:
        - Gestisce fallback a emoji se l'immagine non è disponibile
        - Supporta layout verticale (testo sotto l'immagine) o orizzontale
        - Cattura e gestisce eccezioni durante il caricamento dell'immagine
    """
    try:
        if text_below:
            # Layout verticale con testo sotto l'immagine
            container = st.container()
            img_col = container.container()

            # Immagine
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img_col.image(img, width=img_width, use_container_width=False)
            else:
                # Fallback emoji
                fallback_emojis = {
                    "vegan.png": "🌱",
                    "vegetarian.png": "🥗",
                    "gluten_free.png": "🌾",
                    "lactose_free.png": "🥛"
                }
                emoji = fallback_emojis.get(os.path.basename(img_path), "⚠️")
                img_col.markdown(
                    f"<h1 style='text-align: center;'>{emoji}</h1>", unsafe_allow_html=True)

            # Testo e Checkbox centrati
            return img_col.checkbox(label, key=key, value=value, label_visibility="visible")
        else:
            # Layout originale con colonne affiancate
            col1, col2 = st.columns([1, 10])
            if os.path.exists(img_path):
                img = Image.open(img_path)
                col1.image(img, width=img_width)
            else:
                # Fallback emoji
                fallback_emojis = {
                    "vegan.png": "🌱",
                    "vegetarian.png": "🥗",
                    "gluten_free.png": "🌾",
                    "lactose_free.png": "🥛"
                }
                emoji = fallback_emojis.get(os.path.basename(img_path), "⚠️")
                col1.markdown(
                    f"<h3 style='margin: 0; padding: 0;'>{emoji}</h3>", unsafe_allow_html=True)
            return col2.checkbox(label, key=key, value=value)
    except Exception as e:
        st.error(
            f"Errore nella visualizzazione del checkbox con immagine: {e}")
        return st.checkbox(label, key=key, value=value)
# Function to get image HTML for output formatting


def get_img_html(img_path, width=24):
    """
    Genera un tag HTML img con dati immagine codificati in base64.

    Questa funzione crea un tag HTML per un'immagine che può essere incorporato
    direttamente nell'output HTML, senza necessità di file esterni.

    Args:
        img_path: Percorso all'immagine
        width: Larghezza desiderata dell'immagine in pixel

    Returns:
        str: Tag HTML completo con l'immagine codificata in base64

    Note:
        - Usa la codifica base64 per incorporare l'immagine direttamente nel HTML
        - Fornisce un simbolo di avviso (⚠️) come fallback se l'immagine non è trovata
        - Include stile CSS inline per allineamento e spaziatura
    """
    try:
        if os.path.exists(img_path):
            base64_image = get_base64_encoded_image(img_path)
            return f'<img src="data:image/png;base64,{base64_image}" width="{width}" style="margin-right: 5px; vertical-align: middle;">'
        else:
            return "⚠️"  # Return a warning symbol if image not found
    except Exception as e:
        st.error(f"Error creating image HTML: {e}")
        return "⚠️"


# forse dovrei spostare le funzioni sopra in utils 🚨
# ----- INTERFACCIA PRINCIPALE STREAMLIT -----
st.title("Generatore di Ricette Personalizzate")

# Selettore per il target di carboidrati
target_cho = st.number_input("🎯 Target CHO (g)", min_value=10, value=80)
# Layout con colonne per le icone allineate orizzontalmente
# Layout con colonne per le icone allineate orizzontalmente
st.write("Preferenze dietetiche:")
col1, col2, col3, col4 = st.columns(4)
# Checkbox personalizzati con immagini per le preferenze dietetiche
with col1:
    vegan = image_checkbox("Vegano", vegan_img_path,
                           key="vegan", text_below=True, value=False)

with col2:
    vegetarian = image_checkbox(
        "Vegetariano", vegetarian_img_path, key="vegetarian", text_below=True, value=False)

with col3:
    gluten_free = image_checkbox(
        "Senza Glutine", gluten_free_img_path, key="gluten_free", text_below=True, value=False)

with col4:
    lactose_free = image_checkbox(
        "Senza Lattosio", lactose_free_img_path, key="lactose_free", text_below=True, value=False)
# Pulsante per avviare la generazione delle ricette
if st.button("Genera Ricette"):
    # Mostra gif/indicatore durante la generazione
    gif_container = st.empty()
    if os.path.exists(loading_img_path):
        gif_container.image(loading_img_path)
    else:
        gif_container.info("Elaborazione in corso...")

    start_time = time.time()

    # Crea un dizionario con HTML per le icone da passare all'agente formatter
    img_dict = {
        "vegan": get_img_html(vegan_img_path),
        "vegetarian": get_img_html(vegetarian_img_path),
        "gluten_free": get_img_html(gluten_free_img_path),
        "lactose_free": get_img_html(lactose_free_img_path),
    }

    # Esegui la generazione delle ricette
    output = run_recipe_generation(
        target_cho=target_cho,
        vegan=vegan,
        vegetarian=vegetarian,
        gluten_free=gluten_free,
        lactose_free=lactose_free,
        streamlit_output=True,  # Indica che siamo in Streamlit
        streamlit_write=st.write,
        streamlit_info=st.info,
        streamlit_error=st.error,
        img_dict=img_dict  # Passa il dizionario delle icone
    )

    end_time = time.time()
    generation_time = end_time - start_time
    # Rimuovo la GIF/indicatore dopo la generazione
    gif_container.empty()

    # Visualizza il risultato
    st.header("Ricette Generate")
    st.markdown(output, unsafe_allow_html=True)
    st.info(f"Tempo di generazione: {generation_time:.2f} secondi")
