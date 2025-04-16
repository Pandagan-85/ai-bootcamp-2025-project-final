import time
import os
from PIL import Image
import base64
import streamlit as st
from main import run_recipe_generation


# Create a proper path to access static files in Streamlit


def get_base64_encoded_image(image_path):
    """Returns the base64 encoded image as a string"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


# Set the path to your static folder
# You will need to create a folder named 'static' with your images inside
# The static folder should be in the same directory as your Streamlit app
static_folder = "static"

# Ensure the static folder exists
if not os.path.exists(static_folder):
    os.makedirs(static_folder)
    st.warning(
        f"Created '{static_folder}' folder. Please place your icon images in this folder.")

# Image paths (relative to the static folder)
vegan_img_path = os.path.join(static_folder, "vegan.png")
vegetarian_img_path = os.path.join(static_folder, "vegetarian.png")
gluten_free_img_path = os.path.join(static_folder, "gluten_free.png")
lactose_free_img_path = os.path.join(static_folder, "lactose_free.png")
loading_img_path = os.path.join(static_folder, "loading.gif")

# Function for creating custom checkboxes with images


def image_checkbox(label, img_path, img_width=150, key=None, value=False, text_below=True):
    """
    Crea un checkbox personalizzato con un'immagine.

    Args:
        label (str): Il testo da visualizzare.
        img_path (str): Il percorso all'immagine.
        img_width (int, optional): La larghezza desiderata per l'immagine. Default 60.
        key (str, optional): Una chiave univoca per il widget. Default None.
        value (bool, optional): Il valore iniziale del checkbox. Default False.
        text_below (bool, optional): Se True, mostra il testo sotto l'immagine. Default True.

    Returns:
        bool: True se il checkbox è selezionato, False altrimenti.
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
    """Generate an HTML img tag with base64 encoded image data"""
    try:
        if os.path.exists(img_path):
            base64_image = get_base64_encoded_image(img_path)
            return f'<img src="data:image/png;base64,{base64_image}" width="{width}" style="margin-right: 5px; vertical-align: middle;">'
        else:
            return "⚠️"  # Return a warning symbol if image not found
    except Exception as e:
        st.error(f"Error creating image HTML: {e}")
        return "⚠️"


st.title("Generatore di Ricette Personalizzate")


target_cho = st.number_input("🎯 Target CHO (g)", min_value=10, value=80)
# Layout con colonne per le icone allineate orizzontalmente
st.write("Preferenze dietetiche:")
col1, col2, col3, col4 = st.columns(4)

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

if st.button("Genera Ricette"):
    # Show loading gif while generation is in progress
    gif_container = st.empty()
    if os.path.exists(loading_img_path):
        gif_container.image(loading_img_path)
    else:
        gif_container.info("Elaborazione in corso...")

    start_time = time.time()

    # Create a dictionary with image HTML for formatter_agent
    img_dict = {
        "vegan": get_img_html(vegan_img_path),
        "vegetarian": get_img_html(vegetarian_img_path),
        "gluten_free": get_img_html(gluten_free_img_path),
        "lactose_free": get_img_html(lactose_free_img_path),
    }

    # Run the recipe generation
    output = run_recipe_generation(
        target_cho=target_cho,
        vegan=vegan,
        vegetarian=vegetarian,
        gluten_free=gluten_free,
        lactose_free=lactose_free,
        streamlit_output=True,
        streamlit_write=st.write,
        streamlit_info=st.info,
        streamlit_error=st.error,
        img_dict=img_dict  # Pass the image dictionary to main.py
    )

    end_time = time.time()
    generation_time = end_time - start_time

    gif_container.empty()  # Remove the GIF after generation

    # Display the output
    st.header("Ricette Generate")
    st.markdown(output, unsafe_allow_html=True)
    st.info(f"Tempo di generazione: {generation_time:.2f} secondi")
