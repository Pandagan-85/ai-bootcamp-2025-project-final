"""
Debug script to check ingredient database consistency and FAISS index.
"""
import os
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from utils import normalize_name, find_best_match_faiss
from loaders import load_basic_ingredient_info

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INGREDIENTS_FILE = os.path.join(DATA_DIR, "ingredients.csv")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "ingredients.index")
NAME_MAPPING_FILE = os.path.join(DATA_DIR, "ingredient_names.pkl")
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'


def print_separator(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)


# Load resources
print_separator("Loading Resources")
try:
    # Load ingredient data
    ingredient_data = load_basic_ingredient_info(INGREDIENTS_FILE)
    print(f"Loaded {len(ingredient_data)} ingredients from CSV")

    # Load FAISS index
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    print(f"Loaded FAISS index with {faiss_index.ntotal} vectors")

    # Load name mapping
    with open(NAME_MAPPING_FILE, 'rb') as f:
        index_to_name_mapping = pickle.load(f)
    print(f"Loaded name mapping with {len(index_to_name_mapping)} entries")

    # Check consistency
    if faiss_index.ntotal != len(index_to_name_mapping):
        print(
            f"ERROR: FAISS index size ({faiss_index.ntotal}) does not match name mapping size ({len(index_to_name_mapping)})")
except Exception as e:
    print(f"Error loading resources: {e}")
    exit(1)

# Load model
try:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Loaded SentenceTransformer model: {EMBEDDING_MODEL_NAME}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Check for problematic ingredients reported in the logs
print_separator("Checking Problematic Ingredients")
problem_ingredients = ["polpo", "pomodoro", "Ceci",
                       "Olive nere", "rucola", "Pesce spada", "Salmone"]

for ingredient in problem_ingredients:
    normalized = normalize_name(ingredient)
    print(
        f"\nChecking ingredient: '{ingredient}' (normalized: '{normalized}')")

    # Check if in index_to_name_mapping
    in_mapping = normalized in index_to_name_mapping
    if in_mapping:
        mapping_idx = index_to_name_mapping.index(normalized)
        print(f"✓ Found in index_to_name_mapping at index {mapping_idx}")
    else:
        print(f"✗ NOT found in index_to_name_mapping")
        # Try to find close matches
        close_matches = [
            name for name in index_to_name_mapping if normalized in name or name in normalized]
        if close_matches:
            print(f"  Close matches in mapping: {close_matches}")

    # Check if in ingredient_data keys
    in_data = ingredient in ingredient_data
    if in_data:
        print(f"✓ Found in ingredient_data keys")
    else:
        print(f"✗ NOT found in ingredient_data keys")
        # Try with normalized name
        in_data_normalized = normalized in ingredient_data
        if in_data_normalized:
            print(f"  But found with normalized name in ingredient_data")
        else:
            # Try to find close matches
            close_matches = [name for name in ingredient_data.keys()
                             if normalized in normalize_name(name) or normalize_name(name) in normalized]
            if close_matches:
                print(f"  Close matches in ingredient_data: {close_matches}")

    # Try FAISS matching
    match_result = find_best_match_faiss(
        ingredient,
        faiss_index,
        index_to_name_mapping,
        model,
        normalize_name,
        threshold=0.60  # Lower threshold for testing
    )

    if match_result:
        matched_name, score = match_result
        print(f"✓ FAISS match: '{matched_name}' with score {score:.4f}")

        # Check if the matched name is in ingredient_data
        if matched_name in ingredient_data:
            print(f"  ✓ Matched name found in ingredient_data")
        else:
            print(f"  ✗ Matched name NOT found in ingredient_data")
            # Try to find the exact case in ingredient_data
            for name in ingredient_data.keys():
                if normalize_name(name) == normalize_name(matched_name):
                    print(f"    Found variant in ingredient_data: '{name}'")
    else:
        print(f"✗ No FAISS match found above threshold")

# Print overlapping keys analysis
print_separator("Database Consistency Analysis")

# Count ingredients in mapping not in data
mapping_not_in_data = [name for name in index_to_name_mapping
                       if not any(normalize_name(data_key) == normalize_name(name)
                                  for data_key in ingredient_data.keys())]
print(
    f"Ingredients in FAISS mapping not in ingredient_data: {len(mapping_not_in_data)}")
if mapping_not_in_data and len(mapping_not_in_data) < 20:
    print(f"Examples: {mapping_not_in_data}")

# Count ingredients in data not in mapping
normalized_mapping = [normalize_name(name) for name in index_to_name_mapping]
data_not_in_mapping = [key for key in ingredient_data.keys()
                       if normalize_name(key) not in normalized_mapping]
print(
    f"Ingredients in ingredient_data not in FAISS mapping: {len(data_not_in_mapping)}")
if data_not_in_mapping and len(data_not_in_mapping) < 20:
    print(f"Examples: {data_not_in_mapping}")

# Analyze CHO distributions
print_separator("CHO Distribution Analysis")
cho_values = [info.cho_per_100g for info in ingredient_data.values()
              if info.cho_per_100g is not None]
if cho_values:
    print(f"CHO statistics:")
    print(f"  Min: {min(cho_values):.2f}g per 100g")
    print(f"  Max: {max(cho_values):.2f}g per 100g")
    print(f"  Mean: {sum(cho_values)/len(cho_values):.2f}g per 100g")
    print(f"  Median: {sorted(cho_values)[len(cho_values)//2]:.2f}g per 100g")

    # Count high-CHO ingredients
    high_cho = [key for key, info in ingredient_data.items(
    ) if info.cho_per_100g is not None and info.cho_per_100g > 50]
    print(f"\nHigh-CHO ingredients (>50g per 100g): {len(high_cho)}")
    if high_cho and len(high_cho) < 20:
        print(f"Examples: {', '.join(high_cho)}")

    # Count medium-CHO ingredients
    med_cho = [key for key, info in ingredient_data.items(
    ) if info.cho_per_100g is not None and 20 < info.cho_per_100g <= 50]
    print(f"Medium-CHO ingredients (20-50g per 100g): {len(med_cho)}")
    if med_cho and len(med_cho) < 20:
        print(f"Examples: {', '.join(med_cho)}")

print("\nDebugging completed.")
