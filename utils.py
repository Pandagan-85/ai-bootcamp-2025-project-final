# utils.py
import re
import pickle
from typing import List, Dict, Optional, Tuple, Any, Callable
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
# Importa le classi da model_schema se necessario per type hinting
from model_schema import RecipeIngredient, IngredientInfo, CalculatedIngredient, FinalRecipeOption, UserPreferences


def normalize_name(name: str) -> str:
    """Normalizza il nome per il matching (minuscolo, rimuove eccesso spazi)."""
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r'\s+', ' ', name)
    return name


def find_best_match_faiss(
    llm_name: str,
    faiss_index: faiss.Index,
    index_to_name_mapping: List[str],
    model: SentenceTransformer,
    normalize_func: Callable[[str], str],
    threshold: float = 0.65  # Soglia più bassa
) -> Optional[Tuple[str, float]]:
    """Trova il miglior match usando un approccio a più livelli."""

    # PRETRATTAMENTO e NORMALIZZAZIONE
    common_synonyms = {
        "carote": "carota",
        "gamberi": "gambero",
        "couscous": "cuscus",
        "coriandolo": "coriandolo fresco",
        "peperoni": "peperone",
        "feta": "formaggio feta",
        "olive nere": "olive",
        "formaggio halloumi": "halloumi",
        # Aggiungi altri sinonimi
    }

    # Normalizza input
    normalized_llm = normalize_func(llm_name)

    # 1. TENTATIVO 1: Corrispondenza diretta o tramite sinonimo noto
    if normalized_llm in index_to_name_mapping:
        exact_index = index_to_name_mapping.index(normalized_llm)
        return index_to_name_mapping[exact_index], 1.0

    if normalized_llm in common_synonyms:
        synonym = common_synonyms[normalized_llm]
        if synonym in index_to_name_mapping:
            synonym_index = index_to_name_mapping.index(synonym)
            return index_to_name_mapping[synonym_index], 0.95

    # 2. TENTATIVO 2: Matching FAISS standard
    try:
        query_embedding = model.encode(
            [normalized_llm],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')

        # Cerca i top 3 vicini invece di solo 1
        k = 3
        D, I = faiss_index.search(query_embedding, k)

        # Controlla risultati nell'ordine
        for i in range(min(k, I.shape[1])):
            match_index = I[0][i]
            match_score = D[0][i]

            if 0 <= match_index < len(index_to_name_mapping) and match_score >= threshold:
                matched_name = index_to_name_mapping[match_index]
                return matched_name, float(match_score)

        # TENTATIVO 3: Strategie aggiuntive per ingredienti problematici
        # Verifica forme singolare/plurale con soglia ridotta
        if normalized_llm.endswith('i'):  # Possibile plurale in italiano
            singular = normalized_llm[:-1] + 'o'  # es. "gamberi" → "gambero"
            singular_embedding = model.encode(
                [singular],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype('float32')

            D_sing, I_sing = faiss_index.search(singular_embedding, 1)
            if I_sing.size > 0 and D_sing.size > 0:
                match_index = I_sing[0][0]
                match_score = D_sing[0][0]

                if 0 <= match_index < len(index_to_name_mapping) and match_score >= threshold - 0.1:
                    return index_to_name_mapping[match_index], float(match_score)

        return None
    except Exception as e:
        print(
            f"Errore durante la ricerca FAISS avanzata per '{llm_name}': {e}")
        return None

# --- Mantieni le altre funzioni di utilità ---


def calculate_ingredient_cho_contribution(
    ingredients: List[RecipeIngredient],
    ingredient_data: Dict[str, IngredientInfo]
) -> List[CalculatedIngredient]:
    """Calcola i contributi nutrizionali per una lista di ingredienti."""
    calculated_list = []

    # Crea un dizionario case-insensitive per il matching
    lowercase_to_original = {
        name.lower(): name for name in ingredient_data.keys()}

    for ing in ingredients:
        # Cerca l'ingrediente (ignorando case) nel dizionario
        ingredient_key = ing.name
        if ing.name.lower() in lowercase_to_original:
            ingredient_key = lowercase_to_original[ing.name.lower()]

        info = ingredient_data.get(ingredient_key)
        if info:
            cho_per_100g = info.cho_per_100g if info.cho_per_100g is not None else 0.0
            cho_contribution = (cho_per_100g / 100.0) * ing.quantity_g

            # Calcola contributi altri nutrienti (gestendo None)
            calories = (info.calories_per_100g / 100.0) * \
                ing.quantity_g if info.calories_per_100g is not None else None
            protein = (info.protein_g_per_100g / 100.0) * \
                ing.quantity_g if info.protein_g_per_100g is not None else None
            fat = (info.fat_g_per_100g / 100.0) * \
                ing.quantity_g if info.fat_g_per_100g is not None else None
            fiber = (info.fiber_g_per_100g / 100.0) * \
                ing.quantity_g if info.fiber_g_per_100g is not None else None

            calculated_list.append(
                CalculatedIngredient(
                    name=ing.name,
                    quantity_g=ing.quantity_g,
                    cho_per_100g=cho_per_100g,
                    cho_contribution=round(cho_contribution, 2),
                    # Aggiungi valori nutrizionali calcolati
                    calories_per_100g=info.calories_per_100g,
                    calories_contribution=round(
                        calories, 2) if calories is not None else None,
                    protein_g_per_100g=info.protein_g_per_100g,
                    protein_contribution_g=round(
                        protein, 2) if protein is not None else None,
                    fat_g_per_100g=info.fat_g_per_100g,
                    fat_contribution_g=round(
                        fat, 2) if fat is not None else None,
                    fiber_g_per_100g=info.fiber_g_per_100g,
                    fiber_contribution_g=round(
                        fiber, 2) if fiber is not None else None,
                    # Flag dietetici dall'info base
                    is_vegan=info.is_vegan,
                    is_vegetarian=info.is_vegetarian,
                    is_gluten_free=info.is_gluten_free,
                    is_lactose_free=info.is_lactose_free
                )
            )
        else:
            print(
                f"Attenzione (utils): Info ingrediente '{ing.name}' non trovate durante calcolo CHO.")
            # Aggiungi un placeholder
            calculated_list.append(
                CalculatedIngredient(
                    name=f"{ing.name} (Info Mancanti!)",
                    quantity_g=ing.quantity_g,
                    cho_contribution=0.0
                )
            )

    return calculated_list


def check_dietary_match(recipe: Any, preferences: UserPreferences) -> bool:
    """Verifica se una ricetta (vecchio tipo Recipe) soddisfa le preferenze."""
    # Mantieni questa funzione se ancora usata da qualche parte, altrimenti rimuovi
    if preferences.vegan and not recipe.is_vegan_recipe:
        return False
    if preferences.vegetarian and not recipe.is_vegetarian_recipe:
        return False
    if preferences.gluten_free and not recipe.is_gluten_free_recipe:
        return False
    if preferences.lactose_free and not recipe.is_lactose_free_recipe:
        return False
    return True


def check_final_recipe_dietary_match(recipe: FinalRecipeOption, preferences: UserPreferences) -> bool:
    """Verifica se una ricetta (FinalRecipeOption) soddisfa le preferenze."""
    if preferences.vegan and not recipe.is_vegan:
        return False
    if preferences.vegetarian and not recipe.is_vegetarian:
        return False
    if preferences.gluten_free and not recipe.is_gluten_free:
        return False
    if preferences.lactose_free and not recipe.is_lactose_free:
        return False
    return True
