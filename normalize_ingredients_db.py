"""
Script di utilità per normalizzare i nomi degli ingredienti nel database.

Questo script:
1. Carica il database degli ingredienti dal CSV
2. Crea un backup del file originale
3. Normalizza ogni nome di ingrediente (minuscolo, rimuove spazi in eccesso, prima lettera maiuscola)
4. Rimuove eventuali duplicati creati dalla normalizzazione
5. Salva sia un nuovo file normalizzato che sovrascrive l'originale

Utilizzo:
python normalize_ingredients_db.py

Note:
- Assicurarsi che il file ingredients.csv esista nella cartella data/
- Controlla attentamente l'output per eventuali avvisi di duplicati rimossi
"""
import os
import pandas as pd
from utils import normalize_name

# Configurazione
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INGREDIENTS_FILE = os.path.join(DATA_DIR, "ingredients.csv")
NORMALIZED_INGREDIENTS_FILE = os.path.join(
    DATA_DIR, "ingredients_normalized.csv")

print(f"Caricamento ingredienti da {INGREDIENTS_FILE}...")

try:
    # Carica il CSV degli ingredienti
    df = pd.read_csv(INGREDIENTS_FILE, encoding='utf-8')

    # Backup del CSV originale
    backup_file = os.path.join(DATA_DIR, "ingredients_original_backup.csv")
    print(f"Creazione backup in {backup_file}...")
    df.to_csv(backup_file, index=False)

    # Crea un dizionario di conversione (nome originale -> nome normalizzato)
    original_names = df['name'].tolist()
    print(f"Trovati {len(original_names)} nomi di ingredienti")

    conversion_dict = {}
    for name in original_names:
        if not isinstance(name, str):
            continue
        # Usa la prima lettera maiuscola per uniformità
        normalized = normalize_name(name).capitalize()
        conversion_dict[name] = normalized

    # Applica la normalizzazione alla colonna 'name'
    df['name'] = df['name'].map(lambda x: conversion_dict.get(
        x, x) if isinstance(x, str) else x)

    # Rimuovi eventuali duplicati che potrebbero essere stati creati
    duplicate_mask = df.duplicated(subset=['name'], keep='first')
    if duplicate_mask.any():
        num_duplicates = duplicate_mask.sum()
        print(
            f"Attenzione: Rimozione di {num_duplicates} righe duplicate dopo normalizzazione")
        df = df[~duplicate_mask]

    # Salva il CSV normalizzato
    print(f"Salvataggio CSV normalizzato in {NORMALIZED_INGREDIENTS_FILE}...")
    df.to_csv(NORMALIZED_INGREDIENTS_FILE, index=False)

    # Sovrascrivi il CSV originale (opzionale)
    print(f"Sovrascrittura CSV originale con versione normalizzata...")
    df.to_csv(INGREDIENTS_FILE, index=False)

    print("Normalizzazione completata con successo!")

except Exception as e:
    print(f"Errore durante la normalizzazione: {e}")
