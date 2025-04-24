"""
Funzioni per la conversione e il download delle ricette.

Questo modulo contiene funzioni per convertire l'output HTML delle ricette
in un formato testuale per il download, utilizzando approcci ottimizzati
per evitare ricaricamenti della pagina in Streamlit.
"""
import streamlit as st
from bs4 import BeautifulSoup
import base64
from io import BytesIO
import time


def convert_html_to_text(html_content):
    """
    Converte il contenuto HTML delle ricette in testo formattato.

    Args:
        html_content: Contenuto HTML delle ricette

    Returns:
        Testo formattato delle ricette
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Dizionario per convertire icone in descrizioni testuali invece di emoji
    icon_to_text = {
        "vegan": "[VEGANA]",
        "vegetarian": "[VEGETARIANA]",
        "gluten_free": "[SENZA GLUTINE]",
        "lactose_free": "[SENZA LATTOSIO]"
    }

    # Sostituisci le icone con testo normale
    for img in soup.find_all('img'):
        img_alt = img.get('alt', '').lower()
        replaced = False
        for icon_type, text_replacement in icon_to_text.items():
            if icon_type in img_alt:
                img.replace_with(text_replacement)
                replaced = True
                break
        if not replaced:
            img.replace_with('')

    # Converti HTML in testo con formattazione semplice
    text_content = ""

    # Titolo principale
    if soup.h1:
        text_content += f"{soup.h1.text}\n{'=' * len(soup.h1.text)}\n\n"

    # Descrizione iniziale (paragrafi prima del primo <hr>)
    intro_paragraphs = []
    current = soup.find('h1').find_next_sibling(
    ) if soup.find('h1') else soup.find('p')
    while current and current.name != 'hr':
        if current.name == 'p':
            intro_paragraphs.append(current.text.strip())
        current = current.find_next_sibling()

    if intro_paragraphs:
        text_content += "\n".join(intro_paragraphs) + "\n\n"
        text_content += "-" * 60 + "\n\n"

    # Processa ogni ricetta (definita tra <hr> tags)
    recipes = []
    hr_tags = soup.find_all('hr')

    if hr_tags:
        for i in range(len(hr_tags)):
            recipe_content = ""
            current = hr_tags[i].find_next_sibling()
            while current and (i == len(hr_tags) - 1 or current != hr_tags[i+1]):
                if current.name == 'h2':  # Nome ricetta
                    recipe_content += f"{current.text}\n{'-' * len(current.text)}\n\n"
                elif current.name == 'p':  # Descrizione
                    recipe_content += f"{current.text}\n\n"
                # Sezioni (nutrizione, ingredienti, ecc.)
                elif current.name == 'h3':
                    recipe_content += f"{current.text}:\n"
                elif current.name == 'div' and 'Caratteristiche' in current.text:
                    recipe_content += f"Caratteristiche: {current.text.replace('Caratteristiche:', '').strip()}\n\n"
                # Liste non ordinate (ingredienti, nutrizione)
                elif current.name == 'ul':
                    for li in current.find_all('li'):
                        recipe_content += f"* {li.text.strip()}\n"
                    recipe_content += "\n"
                elif current.name == 'ol':  # Liste ordinate (istruzioni)
                    for idx, li in enumerate(current.find_all('li')):
                        recipe_content += f"{idx+1}. {li.text.strip()}\n"
                    recipe_content += "\n"

                # Interrompi se abbiamo raggiunto un hr o la fine del contenuto
                next_sibling = current.find_next_sibling()
                if next_sibling is None or (i < len(hr_tags) - 1 and next_sibling == hr_tags[i+1]):
                    break

                current = next_sibling

            if recipe_content.strip():
                recipes.append(recipe_content)
    else:
        # Estrazione alternativa se non ci sono tag hr
        for h2 in soup.find_all('h2'):
            recipe_content = f"{h2.text}\n{'-' * len(h2.text)}\n\n"
            current = h2.find_next_sibling()
            while current and current.name != 'h2':
                if current.name == 'p':  # Descrizione
                    recipe_content += f"{current.text}\n\n"
                elif current.name == 'h3':  # Sezioni
                    recipe_content += f"{current.text}:\n"
                elif current.name == 'div' and 'Caratteristiche' in current.text:
                    recipe_content += f"Caratteristiche: {current.text.replace('Caratteristiche:', '').strip()}\n\n"
                elif current.name == 'ul':  # Liste non ordinate
                    for li in current.find_all('li'):
                        recipe_content += f"* {li.text.strip()}\n"
                    recipe_content += "\n"
                elif current.name == 'ol':  # Liste ordinate
                    for idx, li in enumerate(current.find_all('li')):
                        recipe_content += f"{idx+1}. {li.text.strip()}\n"
                    recipe_content += "\n"

                current = current.find_next_sibling()

            if recipe_content.strip() and len(recipe_content) > len(h2.text) + 10:  # Evita ricette vuote
                recipes.append(recipe_content)

    # Aggiungi ricette al testo finale
    text_content += "\n\n".join(recipes)

    # Aggiungi suggerimenti finali se presenti
    suggestions = soup.find('h3', string='Suggerimenti')
    if suggestions:
        text_content += "\nSUGGERIMENTI\n" + "-" * 12 + "\n"
        current = suggestions.find_next_sibling()
        while current and current.name == 'p':
            text_content += f"{current.text}\n"
            current = current.find_next_sibling()

    # Aggiungi footer
    text_content += "\n\n" + "-" * 60 + "\n"
    text_content += "Generato da NutriCHOice - La scelta intelligente per un'alimentazione su misura"

    return text_content


def get_download_link(text_content, filename="ricette_nutricho.txt"):
    """
    Genera un link HTML per il download del contenuto testuale senza ricaricamento pagina.

    Args:
        text_content: Contenuto testuale da scaricare
        filename: Nome del file da generare

    Returns:
        Stringa HTML con link per download
    """
    # Codifica il contenuto in base64
    b64 = base64.b64encode(text_content.encode()).decode()

    # Formato corretto per i browser moderni
    href = f'data:text/plain;charset=utf-8;base64,{b64}'

    # Crea il link HTML - correzione: rimuovi il timestamp dal nome del file
    download_link = f'<a href="{href}" download="{filename}" class="custom-download-button">📥 Scarica Ricette</a>'

    # Stile per il pulsante
    style = """
    <style>
    .custom-download-button {
        display: inline-block;
        background-color: rgb(255, 75, 75);
        color: white !important;
        padding: 10px 15px;
        border-radius: 4px;
        text-decoration: none;
        font-weight: bold;
        margin: 10px 0;
        text-align: center;
        transition: background-color 0.3s;
    }
    .custom-download-button:hover {
        background-color: rgb(220, 75, 75);
    }
    </style>
    """

    return style + download_link


def add_html_download_button(html_content, container=None):
    """
    Aggiunge un pulsante di download HTML che evita il ricaricamento della pagina.

    Args:
        html_content: Contenuto HTML delle ricette
        container: Opzionale, container Streamlit in cui aggiungere il pulsante
    """
    if container is None:
        container = st

    # Converti HTML in testo
    text_content = convert_html_to_text(html_content)

    # Genera link per download
    download_link = get_download_link(text_content)

    # Mostra il link come HTML
    container.markdown(download_link, unsafe_allow_html=True)


def add_streamlit_download_button(html_content, container=None):
    """
    Aggiunge il pulsante di download standard di Streamlit.
    Questa funzione mantiene l'implementazione originale che potrebbe causare
    ricaricamenti in alcune versioni di Streamlit.

    Args:
        html_content: Contenuto HTML delle ricette
        container: Opzionale, container Streamlit in cui aggiungere il pulsante
    """
    if container is None:
        container = st

    # Converti HTML in testo
    text_content = convert_html_to_text(html_content)

    # Aggiungi pulsante di download
    container.download_button(
        label="📥 Scarica Ricette",
        data=text_content,
        file_name="ricette_nutricho.txt",
        mime="text/plain",
    )


# Funzione wrapper che sceglie il metodo migliore in base all'opzione selezionata
def add_download_button(html_content, container=None, use_html_method=True):
    """
    Aggiunge un pulsante di download utilizzando il metodo più appropriato.

    Args:
        html_content: Contenuto HTML delle ricette
        container: Opzionale, container Streamlit in cui aggiungere il pulsante
        use_html_method: Se True, usa il metodo HTML che evita ricaricamenti
    """
    if use_html_method:
        add_html_download_button(html_content, container)
    else:
        add_streamlit_download_button(html_content, container)
