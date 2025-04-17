# generator_agent.py (CORRETTO PER FAISS)
import json
import random
import os
import time
import pickle  # Potrebbe non servire più qui se il mapping è nello stato
import faiss  # Importa faiss se serve type hint
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

# Import LLM e componenti Langchain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
import numpy as np

# Import modelli Pydantic e schema stato
from pydantic import BaseModel, Field as PydanticField, ValidationError
from pydantic.config import ConfigDict
from model_schema import IngredientInfo, FinalRecipeOption, RecipeIngredient, UserPreferences, GraphState, CalculatedIngredient

# === CORREZIONE IMPORT ===
# Importa la NUOVA funzione di matching basata su FAISS e le altre utilità
from utils import calculate_ingredient_cho_contribution, find_best_match_faiss, normalize_name
# =========================

# --- Modelli Pydantic (GeneratedRecipeOutput come prima) ---


class GeneratedRecipeOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    recipe_name: str = PydanticField(description="Nome della ricetta generata")
    description: Optional[str] = PydanticField(
        None, description="Breve descrizione")
    ingredients: List[Dict[str, Any]] = PydanticField(
        description="Lista ingredienti [{'name': str, 'quantity_g': float}]")
    is_vegan: bool = PydanticField(description="Flag vegano LLM")
    is_vegetarian: bool = PydanticField(description="Flag vegetariano LLM")
    is_gluten_free: bool = PydanticField(description="Flag senza glutine LLM")
    is_lactose_free: bool = PydanticField(
        description="Flag senza lattosio LLM")
    instructions: Optional[List[str]] = PydanticField(
        None, description="Lista istruzioni")
    error: Optional[str] = PydanticField(None, description="Errore LLM")

# --- Funzione estrazione JSON (come prima) ---


def extract_json_from_llm_response(response_str: str) -> dict:
    # ... (stesso codice di prima per estrarre JSON) ...
    if response_str.strip().startswith('{') and response_str.strip().endswith('}'):
        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            pass
    if "```json" in response_str and "```" in response_str:
        try:
            start = response_str.find("```json") + 7
            end = response_str.find("```", start)
            if start > 6 and end > start:
                return json.loads(response_str[start:end].strip())
        except (json.JSONDecodeError, ValueError):
            pass
    try:
        start = response_str.find('{')
        end = response_str.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(response_str[start:end+1])
    except (json.JSONDecodeError, ValueError):
        pass
    raise ValueError("Impossibile estrarre un JSON valido dalla risposta")


# --- Funzione Worker per Generare Singola Ricetta (Usa FAISS) ---
def generate_single_recipe(
    preferences: UserPreferences,
    # Componenti singoli dallo stato
    ingredient_data: Dict[str, IngredientInfo],
    faiss_index: faiss.Index,
    index_to_name_mapping: List[str],
    embedding_model: SentenceTransformer,
    normalize_func: Callable[[str], str],
    # Altri argomenti
    generator_chain: Any,  # Tipo specifico di Langchain se noto
    recipe_index: int
) -> Optional[FinalRecipeOption]:
    """
    Genera una singola ricetta, valida ingredienti con FAISS, corregge flag, calcola nutrienti.
    """
    print(f"Thread: Generazione ricetta #{recipe_index+1}")

    # ----- SEZIONE 1: PREPARAZIONE DATI E PROMPT (Usa argomenti diretti) -----
    if not ingredient_data or not index_to_name_mapping:
        print(
            f"Errore (Ricetta #{recipe_index+1}): Dati ingredienti o mapping nomi mancanti.")
        return None

    # Costruisci stringa preferenze (come prima)
    dietary_preferences = []
    if preferences.vegan:
        dietary_preferences.append("vegano")
    elif preferences.vegetarian:
        dietary_preferences.append("vegetariano")
    if preferences.gluten_free:
        dietary_preferences.append("senza glutine")
    if preferences.lactose_free:
        dietary_preferences.append("senza lattosio")
    dietary_preferences_string = ", ".join(
        dietary_preferences) if dietary_preferences else "nessuna preferenza specifica"

    # Prepara lista ingredienti per prompt (filtrata per dieta)
    valid_ingredients_names_for_prompt = []
    valid_db_keys_for_check = set()
    all_db_names = list(ingredient_data.keys())

    for name in all_db_names:
        info = ingredient_data[name]
        normalized_name = normalize_func(name)  # Aggiungi questa linea

        # Aggiungi questo blocco di debug
        if normalize_func(name) == "polpo" or name.lower() == "polpo":
            print(f"DEBUG - Trovato 'polpo' nel database come '{name}'")
            print(f"DEBUG - Flag ingrediente: vegan={info.is_vegan}, vegetarian={info.is_vegetarian}, "
                  f"gluten_free={info.is_gluten_free}, lactose_free={info.is_lactose_free}")
            print(f"DEBUG - Preferenze utente: vegan={preferences.vegan}, vegetarian={preferences.vegetarian}, "
                  f"gluten_free={preferences.gluten_free}, lactose_free={preferences.lactose_free}")
            # Stampa esplicitamente la condizione che determina se l'ingrediente viene filtrato
            filter_condition = ((preferences.vegan and not info.is_vegan) or
                                (preferences.vegetarian and not info.is_vegetarian) or
                                (preferences.gluten_free and not info.is_gluten_free) or
                                (preferences.lactose_free and not info.is_lactose_free))
            print(f"DEBUG - Condizione di filtro: {filter_condition}")
            # Per vedere quali condizioni specifiche sono vere
            if preferences.vegan and not info.is_vegan:
                print(
                    "DEBUG - Filtrato perché l'utente richiede vegano ma l'ingrediente non lo è")
            if preferences.vegetarian and not info.is_vegetarian:
                print(
                    "DEBUG - Filtrato perché l'utente richiede vegetariano ma l'ingrediente non lo è")
            if preferences.gluten_free and not info.is_gluten_free:
                print(
                    "DEBUG - Filtrato perché l'utente richiede senza glutine ma l'ingrediente non lo è")
            if preferences.lactose_free and not info.is_lactose_free:
                print(
                    "DEBUG - Filtrato perché l'utente richiede senza lattosio ma l'ingrediente non lo è")

        # Filtro dietetico
        if (preferences.vegan and not info.is_vegan) or \
           (preferences.vegetarian and not info.is_vegetarian) or \
           (preferences.gluten_free and not info.is_gluten_free) or \
           (preferences.lactose_free and not info.is_lactose_free):
            continue

        valid_ingredients_names_for_prompt.append(name)
        # CORRETTO: Aggiungi il nome originale, non quello normalizzato
        valid_db_keys_for_check.add(name)

    if not valid_ingredients_names_for_prompt:
        print(
            f"Errore (Ricetta #{recipe_index+1}): Nessun ingrediente valido trovato per preferenze.")
        return None

    # Liste per prompt (come prima)
    valid_ingredients_list_prompt = ", ".join(
        valid_ingredients_names_for_prompt)
    relevant_ingredients_prompt_list = []
    for name in valid_ingredients_names_for_prompt:
        info = ingredient_data[name]
        if info.cho_per_100g is not None:
            ing_desc = f"{name} (CHO: {info.cho_per_100g}g per 100g)"
            relevant_ingredients_prompt_list.append(ing_desc)
    max_ingredients_prompt = 100
    if len(relevant_ingredients_prompt_list) > max_ingredients_prompt:
        random.seed(recipe_index)
        relevant_ingredients_prompt_list = random.sample(
            relevant_ingredients_prompt_list, max_ingredients_prompt)
    ingredients_list_string_prompt = "\n".join(
        [f"- {ing}" for ing in relevant_ingredients_prompt_list])

    # Esempi CHO (come prima)
    high_cho_ingredients = []
    medium_cho_ingredients = []
    for name in valid_ingredients_names_for_prompt:
        info = ingredient_data[name]
        if info.cho_per_100g is not None:
            if info.cho_per_100g > 50:
                high_cho_ingredients.append(
                    f"{name} ({info.cho_per_100g}g CHO)")
            elif info.cho_per_100g > 20:
                medium_cho_ingredients.append(
                    f"{name} ({info.cho_per_100g}g CHO)")
    high_cho_examples = ", ".join(random.sample(
        high_cho_ingredients, min(len(high_cho_ingredients), 10))) if high_cho_ingredients else ""
    medium_cho_examples = ", ".join(random.sample(
        medium_cho_ingredients, min(len(medium_cho_ingredients), 10))) if medium_cho_ingredients else ""

    # ----- SEZIONE 2: GENERAZIONE RICETTA CON RETRY (come prima) -----
    max_retries = 2
    retry_delay = 1

    for attempt in range(max_retries + 1):
        try:
            # Esegui chain LLM
            response_str = generator_chain.invoke({
                "target_cho": preferences.target_cho,
                "recipe_index": recipe_index + 1,
                "dietary_preferences": dietary_preferences_string,
                "ingredients_list": ingredients_list_string_prompt,
                "high_cho_examples": high_cho_examples,
                "medium_cho_examples": medium_cho_examples,
                "valid_ingredients": valid_ingredients_list_prompt
            })

            # ----- SEZIONE 3: PARSING E VALIDAZIONE JSON (come prima) -----
            try:
                llm_output = extract_json_from_llm_response(response_str)
                print(
                    f"DEBUG - JSON estratto da LLM: {json.dumps(llm_output, indent=2, ensure_ascii=False)[:300]}...")

                if "error" in llm_output and len(llm_output) == 1:
                    print(
                        f"Thread: Errore LLM ricetta #{recipe_index+1}: {llm_output['error']}")
                    return None

                # Controlla esplicitamente la struttura degli ingredienti
                if "ingredients" in llm_output and llm_output["ingredients"]:
                    print(
                        f"DEBUG - Struttura primo ingrediente: {json.dumps(llm_output['ingredients'][0], indent=2)}")
                else:
                    print(f"DEBUG - Chiave 'ingredients' mancante o vuota nel JSON!")

                validated_output = GeneratedRecipeOutput.model_validate(
                    llm_output)
                if validated_output.error:
                    print(
                        f"Thread: Errore LLM ricetta #{recipe_index+1}: {validated_output.error}")
                    return None

                # ----- SEZIONE 4 & 5: VALIDAZIONE INGREDIENTI CON FAISS E CHECK DIETETICO -----
                invalid_ingredient_details = []
                current_recipe_ingredients_db = []
                actual_matched_db_names = set()

                print(
                    f"Thread: Validazione ingredienti FAISS per ricetta '{validated_output.recipe_name}'...")
                for ing_llm in validated_output.ingredients:
                    ing_name_llm = ing_llm.get("name")
                    if not ing_name_llm:
                        invalid_ingredient_details.append(
                            "Ingrediente senza nome")
                        continue

                    # === CORREZIONE CHIAMATA ===
                    # Chiama la funzione di matching FAISS importata correttamente
                    match_result = find_best_match_faiss(
                        llm_name=ing_name_llm,
                        faiss_index=faiss_index,
                        index_to_name_mapping=index_to_name_mapping,
                        model=embedding_model,
                        normalize_func=normalize_func,
                        threshold=0.55  # Regola se necessario
                    )
                    # ==========================

                    if match_result:
                        matched_db_name, match_score = match_result
                        # Trovato nel DB via FAISS

                        # Trova il nome originale nel database con case sensitivity corretta
                        original_db_name = None
                        for db_name in all_db_names:
                            if normalize_func(db_name).lower() == matched_db_name.lower():
                                original_db_name = db_name
                                break

                        if not original_db_name:
                            detail = f"'{ing_name_llm}' matchato FAISS a '{matched_db_name}' ma non trovato con case originale."
                            print(f"Warning: {detail}")
                            invalid_ingredient_details.append(detail)
                            continue

                        # Ora usa original_db_name (con la capitalizzazione corretta) per accedere al database
                        info = ingredient_data.get(original_db_name)
                        if info is None:
                            detail = f"'{ing_name_llm}' matchato FAISS a '{original_db_name}' ma info non trovate nel database."
                            print(f"Warning: {detail}")
                            invalid_ingredient_details.append(detail)
                            continue

                        # Verifica preferenze dietetiche (se necessario)
                        if ((preferences.vegan and not info.is_vegan) or
                            (preferences.vegetarian and not info.is_vegetarian) or
                            (preferences.gluten_free and not info.is_gluten_free) or
                                (preferences.lactose_free and not info.is_lactose_free)):
                            detail = f"'{ing_name_llm}' matchato FAISS a '{original_db_name}' ma non rispetta preferenze dietetiche."
                            print(f"Warning: {detail}")
                            invalid_ingredient_details.append(detail)
                            continue

                        # Aggiungi questo prima di tentare di accedere a quantity_g
                        print(
                            f"DEBUG - Chiavi disponibili in ing_llm: {list(ing_llm.keys())}")
                        print(f"DEBUG - Contenuto dell'ingrediente: {ing_llm}")

                        try:
                            # Controlla prima se la chiave esiste e prova alternative
                            if "quantity_g" in ing_llm:
                                quantity = float(ing_llm["quantity_g"])
                            elif "quantity" in ing_llm:
                                quantity = float(ing_llm["quantity"])
                            elif "amount" in ing_llm:
                                quantity = float(ing_llm["amount"])
                            elif "amount_g" in ing_llm:
                                quantity = float(ing_llm["amount_g"])
                            else:
                                raise KeyError(
                                    f"Nessuna chiave di quantità trovata. Chiavi disponibili: {list(ing_llm.keys())}")

                            current_recipe_ingredients_db.append(
                                RecipeIngredient(name=matched_db_name, quantity_g=quantity))
                            actual_matched_db_names.add(matched_db_name)
                        except (KeyError, ValueError, TypeError) as e:
                            detail = f"Formato quantità errato per '{ing_name_llm}' (match FAISS: '{matched_db_name}'): {e}"
                            print(f"Warning: {detail}")
                            invalid_ingredient_details.append(detail)
                    else:
                        # Non Trovato via FAISS
                        detail = f"Ingrediente LLM '{ing_name_llm}' non trovato via FAISS con soglia sufficiente."
                        print(f"Warning: {detail}")
                        invalid_ingredient_details.append(detail)

                # Gestisci retry/scarto (come prima)
                if invalid_ingredient_details:
                    print(
                        f"Thread: Ricetta #{recipe_index+1} '{validated_output.recipe_name}' contiene ingredienti non validi/mappati FAISS: {'; '.join(invalid_ingredient_details)}. Retry.")
                    if attempt < max_retries:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        print(
                            f"Thread: Troppi tentativi. Ricetta FAISS '{validated_output.recipe_name}' scartata.")
                        return None

                # Ricalcola flag dietetici (come prima)
                calculated_is_vegan = True
                calculated_is_vegetarian = True
                calculated_is_gluten_free = True
                calculated_is_lactose_free = True

                for db_name in actual_matched_db_names:
                    try:
                        info = ingredient_data[db_name]
                        if not info.is_vegan:
                            calculated_is_vegan = False
                        if not info.is_vegetarian:
                            calculated_is_vegetarian = False
                        if not info.is_gluten_free:
                            calculated_is_gluten_free = False
                        if not info.is_lactose_free:
                            calculated_is_lactose_free = False
                    except KeyError:
                        print(
                            f"Errore interno: Nome ingrediente FAISS '{db_name}' non trovato in data durante ricalcolo flag.")
                        calculated_is_vegan = False
                        calculated_is_vegetarian = False
                        calculated_is_gluten_free = False
                        calculated_is_lactose_free = False
                        break

                corrected_is_vegan = calculated_is_vegan
                corrected_is_vegetarian = calculated_is_vegetarian
                corrected_is_gluten_free = calculated_is_gluten_free
                corrected_is_lactose_free = calculated_is_lactose_free

                # ----- SEZIONE 6: CALCOLO CONTRIBUTI NUTRIZIONALI (come prima) -----
                calculated_ingredients_list = calculate_ingredient_cho_contribution(
                    current_recipe_ingredients_db, ingredient_data
                )
                # ... (calcolo totali come prima) ...
                total_cho = round(sum(
                    ing.cho_contribution for ing in calculated_ingredients_list if ing.cho_contribution is not None), 2)
                total_calories = round(sum(ing.calories_contribution for ing in calculated_ingredients_list if ing.calories_contribution is not None), 2) if any(
                    ing.calories_contribution is not None for ing in calculated_ingredients_list) else None
                total_protein = round(sum(ing.protein_contribution_g for ing in calculated_ingredients_list if ing.protein_contribution_g is not None), 2) if any(
                    ing.protein_contribution_g is not None for ing in calculated_ingredients_list) else None
                total_fat = round(sum(ing.fat_contribution_g for ing in calculated_ingredients_list if ing.fat_contribution_g is not None), 2) if any(
                    ing.fat_contribution_g is not None for ing in calculated_ingredients_list) else None
                total_fiber = round(sum(ing.fiber_contribution_g for ing in calculated_ingredients_list if ing.fiber_contribution_g is not None), 2) if any(
                    ing.fiber_contribution_g is not None for ing in calculated_ingredients_list) else None

                # ----- SEZIONE 7: COSTRUISCI OGGETTO FINALE (come prima) -----
                final_recipe = FinalRecipeOption(
                    name=validated_output.recipe_name,
                    description=validated_output.description,
                    ingredients=calculated_ingredients_list,
                    total_cho=total_cho,
                    total_calories=total_calories,
                    total_protein_g=total_protein,
                    total_fat_g=total_fat,
                    total_fiber_g=total_fiber,
                    is_vegan=corrected_is_vegan,
                    is_vegetarian=corrected_is_vegetarian,
                    is_gluten_free=corrected_is_gluten_free,
                    is_lactose_free=corrected_is_lactose_free,
                    instructions=validated_output.instructions
                )

                print(
                    f"Thread: Ricetta FAISS #{recipe_index+1} '{final_recipe.name}' generata e validata (CHO: {total_cho}g).")
                return final_recipe  # Successo!

            # Gestione errori JSON/Validazione (come prima)
            except (json.JSONDecodeError, ValidationError, KeyError, ValueError) as json_error:
                if attempt < max_retries:
                    print(
                        f"Thread: Errore JSON/Validazione FAISS ricetta #{recipe_index+1}: {json_error}. Retry {attempt+1}/{max_retries}")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                print(
                    f"Thread: Errore JSON/Validazione FAISS definitivo ricetta #{recipe_index+1}: {json_error}")
                return None

        # Gestione errori API/Imprevisti (come prima)
        except Exception as e:
            if attempt < max_retries:
                print(
                    f"Thread: Errore API/Imprevisto FAISS ricetta #{recipe_index+1}: {e}. Retry {attempt+1}/{max_retries}")
                time.sleep(retry_delay * (attempt + 1))
                continue
            print(
                f"Thread: Errore API/Imprevisto FAISS definitivo ricetta #{recipe_index+1}: {e}")
            return None

    return None  # Fallimento dopo tutti i tentativi


# --- Funzione Agente Principale (Corretta per recupero stato) ---
def generate_recipes_agent(state: GraphState) -> GraphState:
    """
    Node Function: Genera ricette usando FAISS e ThreadPoolExecutor.
    """
    print("--- ESECUZIONE NODO: Generazione Ricette (FAISS) ---")

    try:
        # --- Recupera tutti i componenti individuali dallo stato ---
        preferences = state['user_preferences']
        available_ingredients_data = state['available_ingredients_data']
        embedding_model = state['embedding_model']
        normalize_function = state['normalize_function']
        faiss_index = state['faiss_index']
        index_to_name_mapping = state['index_to_name_mapping']
        # --- Fine Recupero Componenti ---

        # --- Verifica componenti stato ---
        print("DEBUG (Agent FAISS): Verifica componenti stato ricevuti...")
        components_ok = True
        error_details = []
        if not preferences:
            components_ok = False
            error_details.append("preferences")
        if not available_ingredients_data:
            components_ok = False
            error_details.append("available_ingredients_data")
        if not embedding_model:
            components_ok = False
            error_details.append("embedding_model")
        if not normalize_function:
            components_ok = False
            error_details.append("normalize_function")
        if not faiss_index:
            components_ok = False
            error_details.append("faiss_index")
        if not index_to_name_mapping:
            components_ok = False
            error_details.append("index_to_name_mapping")
        # Verifica consistenza indice/mapping
        len_map = len(
            index_to_name_mapping) if index_to_name_mapping is not None else 0
        idx_total = faiss_index.ntotal if faiss_index is not None else -1
        if not (len_map > 0 and idx_total > 0 and len_map == idx_total):
            components_ok = False
            error_details.append(
                f"Mismatch/Empty Faiss/Mapping (idx:{idx_total} vs map:{len_map})")

        print(f"DEBUG (Agent FAISS): Controllo componenti OK: {components_ok}")

        if not components_ok:
            error_msg = f"Dati essenziali mancanti/non validi nello stato (FAISS): {'; '.join(error_details)}."
            print(f"Errore: {error_msg}")
            state['error_message'] = error_msg
            state['generated_recipes'] = []
            return state
        # --- Fine Verifica ---

    except KeyError as e:
        print(
            f"Errore CRITICO (Agent FAISS): Chiave mancante nello stato! Chiave: {e}")
        state['error_message'] = f"Stato ricevuto incompleto, chiave mancante: {e}"
        state['generated_recipes'] = []
        return state
    except Exception as ex:
        print(
            f"Errore imprevisto (Agent FAISS) durante recupero/controllo stato: {ex}")
        import traceback
        traceback.print_exc()
        state['error_message'] = f"Errore imprevisto setup generatore FAISS: {ex}"
        state['generated_recipes'] = []
        return state

    # ----- Setup LLM, Prompt, Workers (come prima) -----
    target_recipes = 10  # Numero ricette da tentare
    max_workers = min(6, target_recipes)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        state['error_message'] = "API Key OpenAI non trovata."
        print(f"Errore: {state['error_message']}")
        state['generated_recipes'] = []
        return state

    model_name = "gpt-3.5-turbo"
    print(f"Utilizzo modello {model_name} per generazione ricette FAISS")
    llm = ChatOpenAI(temperature=0.8, model_name=model_name,
                     openai_api_key=api_key)
    system_prompt = """
    **RUOLO: ** Sei un esperto chef e nutrizionista specializzato nella creazione di ricette complete, realistiche, bilanciate e personalizzate.

    **COMPITO: ** Genera una ricetta originale che soddisfi precisamente le esigenze nutrizionali e le preferenze dietetiche dell'utente, seguendo TUTTE le istruzioni sottostanti.

    ---

    ## 1. OBIETTIVI NUTRIZIONALI OBBLIGATORI

    * **Target Carboidrati (CHO):**
        * La ricetta DEVE contenere **{target_cho}g** di CHO, con una tolleranza massima di **±5g**.
        * *Questo vincolo è FONDAMENTALE e verrà verificato.*
    * **Distribuzione Macronutrienti (Indicativa su Grammi):**
        * Punta a un bilanciamento approssimativo dei **grammi** totali dei macronutrienti:
            * CHO: ~45–60% dei grammi totali di macro
            * Proteine: ~15–20% dei grammi totali di macro
            * Grassi: ~25–30% dei grammi totali di macro
        * *Nota: L'obiettivo primario è il target CHO in grammi.*

    ---

    ## 2. UTILIZZO DEGLI INGREDIENTI

    * **Lista Esclusiva:** Usa **SOLO ED ESCLUSIVAMENTE** gli ingredienti forniti alla fine di questo prompt (`{valid_ingredients}`).
        * *NON inventare o usare ingredienti non presenti in quella lista.*
        * *Considera la lista fornita come l'unica dispensa disponibile.*
    * **Quantità:** Specifica la quantità **ESATTA in grammi** per ogni ingrediente usato.
    * **Limite per Ingrediente:** **NESSUN singolo ingrediente può superare i 200g.** *Questo limite è CRITICO.*
    * **Limite Condimenti:** La somma totale dei condimenti (spezie, erbe, salse concentrate, aceti, sale, ecc.) non deve superare i **10g**.
    * **Restrizione Pasta/Riso:** **NON usare MAI pasta e riso insieme** nella stessa ricetta. Scegline solo uno, se necessario.

    ---

    ## 3. STRUTTURA E BILANCIAMENTO DEL PIATTO

    * **Numero Ingredienti:** Includi almeno **4-5 ingredienti diversi**.
    * **Componenti Essenziali:** Assicurati di includere (compatibilmente con le preferenze dietetiche):
        * Una **fonte proteica principale** (carne, pesce, pollame, uova, legumi, tofu, tempeh, seitan, latticini proteici).
        * Una **fonte di carboidrati** (cereali, patate, pane, legumi, frutta).
        * Almeno una **verdura o frutta**.
        * Una **fonte di grassi sani** (olio EVO, avocado, frutta secca, semi).
    * **Quantità Indicative (Linee Guida Utili):**
        * Fonte di Carboidrati (pasta/riso secchi, patate, pane): ~70–120g (se non diversamente specificato per CHO alti/bassi).
        * Fonte Proteica (carne, pesce, tofu): ~100–200g.
        * Verdure: ~100–200g (se non diversamente specificato per CHO alti/bassi).

    ---

    ## 4. STRATEGIA PER IL TARGET CHO

    1.  **Selezione:** Scegli una combinazione bilanciata di ingredienti ad alto, medio e basso contenuto di CHO dalla lista fornita.
        * *Esempio Alto CHO:* `{high_cho_examples}`
        * *Esempio Medio CHO:* `{medium_cho_examples}`
    2.  **Calcolo:** Calcola con precisione il contributo CHO di ogni ingrediente: `CHO_contributo = (quantità_g * CHO_per_100g) / 100`.
    3.  **Verifica Totale:** Assicurati che la somma dei `CHO_contributo` sia **entro ±5g** dal `{target_cho}` richiesto.
    4.  **Equilibrio:** Mantieni la distribuzione bilanciata dei macronutrienti come indicato al punto 1.

    ---

    ## 5. PREFERENZE DIETETICHE (Obbligatorie)

    * Rispetta **RIGOROSAMENTE** le seguenti preferenze:
        * Vegano: {{is_vegan}}
        * Vegetariano: {{is_vegetarian}}
        * Senza Glutine: {{is_gluten_free}}
        * Senza Lattosio: {{is_lactose_free}}
    * *Usa solo ingredienti che sono compatibili con tutte le preferenze impostate a `true`.*

    ---

    ## 6. CASI SPECIALI (Adattamento della Strategia)


    * **Nessuna Preferenza Specifica:** Se tutti i flag dietetici sono `false`, sei incoraggiato a includere ingredienti di origine animale (carne magra, pesce, pollame, uova, formaggi magri) per creare ricette più varie e bilanciate, sempre rispettando il target CHO.
    * **Target CHO < 20g:** Limita drasticamente o escludi pasta, riso, pane, couscous, patate ad alto CHO. Focalizzati su proteine magre, verdure a basso CHO, grassi sani e latticini (se permessi).
    * **Target CHO tra 40g e 100g:** PREFERISCI FORTEMENTE l'uso della pasta come fonte principale di carboidrati. Utilizza tipi diversi di pasta (spaghetti, penne, fusilli, fettuccine, ecc.) e varia i condimenti per creare ricette diverse. Usa circa 70-120g di pasta secca per porzione.
    * **Target CHO > 100g:** La ricetta DEVE essere basata su porzioni significative di alimenti ad alta densità di CHO, con FORTE PREFERENZA PER LA PASTA. Usa le seguenti quantità come riferimento PRINCIPALE:
        * Pasta (peso secco): ~100–180g
        * Pane: ~150–250g
        * Patate: ~300–500g
        * Legumi (peso secco): ~80–150g
        * *In questo caso:* Limita le verdure totali a max 200–300g e rispetta il limite di 10g per i condimenti.

    ---

    # 7. DIVERSIFICAZIONE (Obbligatoria per Ricette Multiple)

    # {recipe_index}** che stai generando in questa sessione.
    * Questa è la ** Ricetta
    * **Se Ricetta  # 1:** Massima libertà nella scelta.
    * **Se Ricetta  # 2 o Successiva:**
        * **DEVI ** creare una ricetta ** COMPLETAMENTE DIVERSA ** dalle precedenti.
        * **Cambia: ** Nome, concetto del piatto, ingredienti principali, stile di cucina(es. Mediterranea, Asiatica, Messicana...), tipologia(primo, secondo, piatto unico, zuppa, bowl...), tecnica di cottura(forno, padella, vapore, griglia, crudo...).
        * **NON RIPETERE ** gli stessi ingredienti principali(soprattutto fonte proteica e fonte CHO principale).

    ---

    # 8. FORMATO OUTPUT (Obbligatorio)

    * Fornisci la ricetta ** ESCLUSIVAMENTE ** nel seguente formato JSON, senza alcun testo o commento aggiuntivo prima o dopo:

    ```json
    {{
    "recipe_name": "Nome Creativo e Unico della Ricetta",
    "description": "Breve descrizione accattivante del piatto",
    "ingredients": [
        {{"name": "Nome Ingrediente Valido 1", "quantity_g": 120.0}},  // DEVE usare esattamente "quantity_g"
        {{"name": "Nome Ingrediente Valido 2", "quantity_g": 85.5}}    // DEVE usare esattamente "quantity_g"
    // ... altri ingredienti ...
    ],
    "is_vegan": boolean, // Deve riflettere la ricetta finale
    "is_vegetarian": boolean, // Deve riflettere la ricetta finale
    "is_gluten_free": boolean, // Deve riflettere la ricetta finale
    "is_lactose_free": boolean, // Deve riflettere la ricetta finale
    "instructions": [
        "Passo 1: Istruzione chiara e dettagliata.",
        "Passo 2: Altra istruzione chiara."
        // ... altri passi ...
    ]
    }}
    ```
    """  # Fine system_prompt

    human_prompt = """
    # INGREDIENTI DISPONIBILI (Usa solo questi):
    {valid_ingredients}

    Genera la ricetta #{recipe_index} rispettando questi criteri:

    TARGET NUTRIZIONALE:
    - Carboidrati(CHO): {target_cho}g(±5g) - QUESTO È CRUCIALE!
    - Distribuzione: Bilanciata, seguendo le preferenze dell'utente, tentando di include proteine, carboidarti e verdure.
    - IMPORTANTE: Nessun ingrediente deve superare i 200g.
    - IMPORTANTE: Nelle ricette non devi usare pasta e riso insieme, ma solo uno dei due. Ad esempio non puoi proporre "Pasta e riso".

    PREFERENZE DIETETICHE:
    {dietary_preferences}
    (Ricorda se l'utente non ne specifica nessuna, sei incoraggiato a includere ingredienti animali (es. carne magra, pollame, pesce, uova, formaggi magri)

    INGREDIENTI DISPONIBILI DETTAGLIATI (con valori nutrizionali per 100g):
    {ingredients_list}

    LISTA COMPLETA DI INGREDIENTI VALIDI:
    IMPORTANTE: Usa ESCLUSIVAMENTE questi ingredienti (sono gli unici disponibili nella dispensa):
    {valid_ingredients}

    Crea una ricetta bilanciata, gustosa e originale che soddisfi ESATTAMENTE il target CHO richiesto.
    Calcola attentamente i CHO totali prima di finalizzare e assicurati che rientrino nel range {target_cho}g ±5g.
    Fornisci l'output esclusivamente in formato JSON come specificato, senza commenti aggiuntivi.

    IMPORTANTE: Questa è la ricetta #{recipe_index}. Assicurati che sia COMPLETAMENTE DIVERSA dalle ricette precedenti nel nome, concetto e stile del piatto.
    Se stai creando:
    - Ricetta #1-2: Scegli liberamente.
    - Ricetta #3-4: Scegli un tipo di cucina e un metodo di cottura diversi dai precedenti.
    - Ricetta #5+: Crea un piatto di un'altra cultura culinaria (mediterranea, asiatica, americana, ecc.) non ancora utilizzata.
    """  # Fine human_prompt

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", human_prompt)])
    generator_chain = prompt | llm | StrOutputParser()

    print(
        f"Avvio generazione FAISS di {target_recipes} ricette con {max_workers} worker paralleli...")
    generated_recipes_list: List[FinalRecipeOption] = []
    recipe_names = set()

    # ----- Esecuzione Parallela (PASSANDO I COMPONENTI CORRETTI) -----
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                generate_single_recipe,  # Funzione worker
                preferences,            # Preferenze
                available_ingredients_data,  # Dati ingredienti (dict)
                faiss_index,                # Indice FAISS
                index_to_name_mapping,      # Mapping nomi
                embedding_model,            # Modello SBERT
                normalize_function,         # Funzione normalize
                generator_chain,            # Chain LLM
                i                           # Indice ricetta
            ) for i in range(target_recipes)
        ]
        # ----- Raccolta Risultati (come prima) -----
        for future in futures:
            try:
                result = future.result()
                if result:
                    if result.name not in recipe_names:
                        recipe_names.add(result.name)
                        generated_recipes_list.append(result)
                        print(f"Ricetta FAISS '{result.name}' aggiunta.")
                    else:
                        print(
                            f"Ricetta FAISS '{result.name}' scartata (duplicata).")
            except Exception as exc:
                print(
                    f"Generazione ricetta FAISS fallita con eccezione: {exc}")

    # ----- Logica Generazione Aggiuntiva (come prima) -----
    if len(generated_recipes_list) < 3:  # Usa la lista appena popolata
        remaining_to_generate = max(3 - len(generated_recipes_list), 2)
        print(
            f"Generazione FAISS di {remaining_to_generate} ricette aggiuntive...")
        llm_diverse = ChatOpenAI(
            temperature=0.95, model_name=model_name, openai_api_key=api_key)
        diverse_system_prompt = system_prompt + \
            "\n\nNOTA SPECIALE: DIVERSITÀ ASSOLUTA RICHIESTA."
        diverse_prompt = ChatPromptTemplate.from_messages(
            [("system", diverse_system_prompt), ("human", human_prompt)])
        generator_chain_diverse = diverse_prompt | llm_diverse | StrOutputParser()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            additional_futures = [
                executor.submit(
                    generate_single_recipe,  # Stessa funzione worker
                    preferences, available_ingredients_data, faiss_index, index_to_name_mapping,
                    embedding_model, normalize_function,  # Passa tutti gli argomenti
                    generator_chain_diverse, i + target_recipes
                ) for i in range(remaining_to_generate)
            ]
            for future in additional_futures:
                try:
                    result = future.result()
                    if result and result.name not in recipe_names:
                        recipe_names.add(result.name)
                        generated_recipes_list.append(result)
                        print(
                            f"Ricetta aggiuntiva FAISS '{result.name}' aggiunta.")
                except Exception as exc:
                    print(
                        f"Generazione ricetta aggiuntiva FAISS fallita: {exc}")

    # ----- Filtro Diversità e Aggiornamento Stato (come prima) -----
    final_recipes = generated_recipes_list
    filtered_final_recipes = []
    ingredient_sets = []
    # Assicurati che CalculatedIngredient abbia 'cho_contribution' o adatta qui
    for recipe in final_recipes:
        main_ingredients = set([ing.name for ing in recipe.ingredients if getattr(
            ing, 'cho_contribution', 0) > 5.0])
        is_unique = True
        for existing_set in ingredient_sets:
            if main_ingredients and existing_set:
                overlap_ratio = len(main_ingredients.intersection(
                    existing_set)) / len(main_ingredients) if len(main_ingredients) > 0 else 0
                if overlap_ratio > 0.6:
                    is_unique = False
                    print(
                        f"Ricetta FAISS '{recipe.name}' scartata per similarità (Overlap: {overlap_ratio:.2f}).")
                    break
        if is_unique:
            ingredient_sets.append(main_ingredients)
            filtered_final_recipes.append(recipe)

    print(
        f"--- Generazione FAISS completata. Ricette uniche generate: {len(filtered_final_recipes)} ---")
    state['generated_recipes'] = filtered_final_recipes
    if not filtered_final_recipes:
        if not state.get('error_message'):
            state['error_message'] = "Nessuna ricetta FAISS valida generata dopo filtri."
    else:
        state.pop('error_message', None)

    return state
