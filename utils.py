import re

from typing import List, Dict, Optional, Tuple, Any, Callable

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
    """
    Trova il miglior match per un nome di ingrediente usando un approccio a più livelli con FAISS.

    Implementa una strategia multi-step:
    1. Tenta corrispondenza diretta con nomi esatti
    2. Tenta match con sinonimi comuni
    3. Esegue ricerca semantica con FAISS
    4. Tenta strategie di fallback (es: forme singolare/plurale)

    Args:
        llm_name: Nome dell'ingrediente generato dall'LLM da matchare
        faiss_index: Indice FAISS precaricato con embeddings degli ingredienti
        index_to_name_mapping: Lista di mapping da indice a nome ingrediente
        model: Modello SentenceTransformer per generare embeddings
        normalize_func: Funzione per normalizzare i nomi degli ingredienti
        threshold: Soglia minima di similarità (default: 0.65)

    Returns:
        Tuple con (nome_matchato, score_similarità) se trovato, None altrimenti
    """

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


def calculate_ingredient_cho_contribution(
    ingredients: List[RecipeIngredient],
    ingredient_data: Dict[str, IngredientInfo]
) -> List[CalculatedIngredient]:
    """Calcola i contributi nutrizionali per una lista di ingredienti.

    Processo:
    1. Per ogni ingrediente, cerca una corrispondenza nel database (exact, case-insensitive, sinonimi)
    2. Calcola i contributi nutrizionali (CHO, calorie, proteine, grassi, fibre) in base alla quantità
    3. Preserva anche i flag dietetici (vegano, senza glutine, ecc.) dall'info base

    Args:
        ingredients: Lista di oggetti RecipeIngredient con nome e quantità in grammi
        ingredient_data: Dizionario con i dati nutrizionali degli ingredienti {nome: IngredientInfo}

    Returns:
        Lista di oggetti CalculatedIngredient con tutti i contributi nutrizionali calcolati
        Gli ingredienti non trovati nel DB vengono comunque inclusi con CHO=0 e un flag "(Info Mancanti!)"""
    calculated_list = []

    # Crea un dizionario case-insensitive per il matching
    lowercase_to_original = {
        normalize_name(name): name for name in ingredient_data.keys()}

    for ing in ingredients:
        # Cerca l'ingrediente (ignorando case) nel dizionario
        ingredient_key = None

        # Match diretto per nome esatto
        if ing.name in ingredient_data:
            ingredient_key = ing.name
        # Match case-insensitive
        elif normalize_name(ing.name) in lowercase_to_original:
            ingredient_key = lowercase_to_original[normalize_name(ing.name)]

        # Se non trovato, prova sinonimi comuni
        if not ingredient_key:
            common_synonyms = {
                "polpo": "polipo",
                "pomodoro": "pomodori",
                "pomodori": "pomodoro",
                "ceci": "cece",
                "olive": "oliva",
                "olive nere": "olive",
                "rucola": "rughetta",
            }

            normalized_name = normalize_name(ing.name)
            if normalized_name in common_synonyms:
                synonym = common_synonyms[normalized_name]
                if synonym in ingredient_data:
                    ingredient_key = synonym
                elif synonym in lowercase_to_original:
                    ingredient_key = lowercase_to_original[synonym]

        # Se ancora non trovato, prova variazioni singolare/plurale
        if not ingredient_key:
            normalized_name = normalize_name(ing.name)

            # Singolare → Plurale
            plural_form = None
            if normalized_name.endswith('o'):
                # es. pomodoro → pomodori
                plural_form = normalized_name[:-1] + 'i'
            elif normalized_name.endswith('a'):
                plural_form = normalized_name[:-1] + 'e'  # es. carota → carote

            if plural_form and plural_form in lowercase_to_original:
                ingredient_key = lowercase_to_original[plural_form]

            # Plurale → Singolare
            if not ingredient_key and (normalized_name.endswith('i') or normalized_name.endswith('e')):
                singular_form = None
                if normalized_name.endswith('i'):
                    # es. pomodori → pomodoro
                    singular_form = normalized_name[:-1] + 'o'
                elif normalized_name.endswith('e'):
                    # es. carote → carota
                    singular_form = normalized_name[:-1] + 'a'

                if singular_form and singular_form in lowercase_to_original:
                    ingredient_key = lowercase_to_original[singular_form]

        # Procedi con il calcolo se l'ingrediente è stato trovato
        if ingredient_key and ingredient_key in ingredient_data:
            info = ingredient_data[ingredient_key]
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
                    is_lactose_free=info.is_lactose_free,
                    # Aggiungi nome originale per il debug
                    original_llm_name=ing.name
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
                    cho_contribution=0.0,
                    original_llm_name=ing.name
                )
            )

    return calculated_list


def check_dietary_match(recipe: Any, preferences: UserPreferences) -> bool:
    """Verifica se una ricetta (vecchio tipo Recipe) soddisfa le preferenze dietetiche dell'utente.

    Args:
        recipe: Oggetto ricetta (vecchio tipo) con flag dietetici is_X_recipe
        preferences: Preferenze dietetiche dell'utente

    Returns:
        True se la ricetta soddisfa tutte le preferenze dietetiche, False altrimenti

    Note:
        Funzione mantenuta per retrocompatibilità. Considerare l'uso di
        check_final_recipe_dietary_match per il nuovo tipo FinalRecipeOption."""
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
    """Verifica se una ricetta (FinalRecipeOption) soddisfa le preferenze dietetiche dell'utente.

    Args:
        recipe: Oggetto FinalRecipeOption con flag dietetici (is_vegan, is_vegetarian, ecc.)
        preferences: Preferenze dietetiche dell'utente (vegan, vegetarian, gluten_free, lactose_free)

    Returns:
        True se la ricetta soddisfa tutte le preferenze dietetiche, False altrimenti"""
    if preferences.vegan and not recipe.is_vegan:
        return False
    if preferences.vegetarian and not recipe.is_vegetarian:
        return False
    if preferences.gluten_free and not recipe.is_gluten_free:
        return False
    if preferences.lactose_free and not recipe.is_lactose_free:
        return False
    return True
