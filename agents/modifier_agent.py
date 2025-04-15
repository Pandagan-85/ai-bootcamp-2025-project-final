# agents/modifier_agent.py

import json
import os
import time
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Usa pydantic_v1 per compatibilità Langchain
from pydantic import BaseModel, Field as PydanticField, ValidationError
from pydantic.config import ConfigDict

from model_schema import GraphState, Recipe, IngredientInfo, FinalRecipeOption, CalculatedIngredient, UserPreferences, RecipeIngredient
from prompts import MODIFIER_SYSTEM_PROMPT
from utils import calculate_total_cho, calculate_ingredient_cho_contribution

# --- Definizione Struttura Output LLM (usando Pydantic v1 per Langchain) ---


class ModifiedRecipeOutput(BaseModel):
    """Struttura Pydantic per l'output JSON atteso dall'LLM Modifier."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    recipe_name: str = PydanticField(
        description="Nome della ricetta modificata o originale")
    final_ingredients: List[Dict[str, Any]] = PydanticField(
        description="Lista di ingredienti finali [{'name': str, 'quantity_g': float}]")
    final_total_cho: float = PydanticField(
        description="CHO totali calcolati per la ricetta finale")
    is_vegan: bool = PydanticField(
        description="Flag vegano per la ricetta finale")
    is_vegetarian: bool = PydanticField(
        description="Flag vegetariano per la ricetta finale")
    is_gluten_free: bool = PydanticField(
        description="Flag senza glutine per la ricetta finale")
    is_lactose_free: bool = PydanticField(
        description="Flag senza lattosio per la ricetta finale")
    error: Optional[str] = PydanticField(
        None, description="Messaggio di errore se la modifica non è possibile")


def extract_json_from_llm_response(response_str: str) -> dict:
    """
    Estrae l'oggetto JSON dalla risposta dell'LLM.
    """
    # Controlla se è già un JSON pulito
    if response_str.strip().startswith('{') and response_str.strip().endswith('}'):
        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            pass

    # Controlla se è racchiuso in markdown code block
    if "```json" in response_str and "```" in response_str:
        try:
            # Estrai il contenuto tra i delimitatori markdown
            start = response_str.find("```json") + 7
            end = response_str.find("```", start)
            if start > 6 and end > start:
                json_content = response_str[start:end].strip()
                return json.loads(json_content)
        except (json.JSONDecodeError, ValueError):
            pass

    # Ultimo tentativo: trovare le parentesi graffe esterne
    try:
        start = response_str.find('{')
        end = response_str.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_content = response_str[start:end+1]
            return json.loads(json_content)
    except (json.JSONDecodeError, ValueError):
        pass

    # Impossibile estrarre JSON valido
    raise ValueError("Impossibile estrarre un JSON valido dalla risposta")


def filter_by_initial_cho(recipes: List[Recipe], ingredient_data: Dict[str, IngredientInfo],
                          target_cho: float, tolerance: float = 15.0) -> List[Recipe]:
    """
    Pre-filtra le ricette in base alla loro vicinanza al target CHO.
    Questo riduce il numero di chiamate API necessarie.
    """
    filtered_recipes = []
    min_cho = target_cho - tolerance
    max_cho = target_cho + tolerance

    for recipe in recipes:
        # Calcola CHO iniziale
        initial_cho = calculate_total_cho(recipe.ingredients, ingredient_data)
        # Se la ricetta è già vicina al target, potrebbe richiedere meno modifiche
        if min_cho <= initial_cho <= max_cho or abs(initial_cho - target_cho) < tolerance * 1.5:
            filtered_recipes.append(recipe)

    # Se il filtro è troppo restrittivo, include alcune ricette aggiuntive
    if len(filtered_recipes) < 10 and len(recipes) > 10:
        # Ordina il resto delle ricette in base alla vicinanza al target CHO
        remaining_recipes = [r for r in recipes if r not in filtered_recipes]
        remaining_recipes.sort(key=lambda r: abs(
            calculate_total_cho(r.ingredients, ingredient_data) - target_cho))
        # Aggiungi le ricette più vicine al target fino ad avere almeno 10 ricette
        filtered_recipes.extend(remaining_recipes[:10-len(filtered_recipes)])

    return filtered_recipes


def process_single_recipe(recipe: Recipe, preferences: UserPreferences,
                          ingredient_data: Dict[str, IngredientInfo],
                          modifier_chain, cho_tolerance: float) -> Optional[FinalRecipeOption]:
    """
    Processa una singola ricetta con l'LLM e restituisce la ricetta modificata.
    Questa funzione viene chiamata in parallelo.
    """
    recipe_name = recipe.name
    print(f"Thread: Elaborazione ricetta: {recipe_name}")

    original_ingredients_str = ", ".join(
        [f"{ing.name} ({ing.quantity_g}g)" for ing in recipe.ingredients])

    # Prepara la rappresentazione del database ingredienti per il prompt
    # Ottimizzazione: includi solo gli ingredienti rilevanti per risparmiare token

    relevant_ingredients = set([ing.name for ing in recipe.ingredients])
    # Aggiungi alcuni alternative dietetiche pertinenti
    if preferences.vegan:
        # Aggiungi alternative vegane comuni
        relevant_ingredients.update([name for name, info in ingredient_data.items()
                                     if info.is_vegan and info.cho_per_100g > 0])
    if preferences.gluten_free:
        # Aggiungi alternative senza glutine
        relevant_ingredients.update([name for name, info in ingredient_data.items()
                                     if info.is_gluten_free and info.cho_per_100g > 0])
    if preferences.lactose_free:
        # Aggiungi alternative senza lattosio
        relevant_ingredients.update([name for name, info in ingredient_data.items()
                                     if info.is_lactose_free and info.cho_per_100g > 0])

    # Limita a max 30 ingredienti per ridurre dimensione prompt
    if len(relevant_ingredients) > 30:
        # Mantieni originali e seleziona un sottoinsieme
        original_names = [ing.name for ing in recipe.ingredients]
        others = list(relevant_ingredients - set(original_names))
        # Seleziona un numero di ingredienti extra per arrivare a max 30
        selected = original_names + others[:30-len(original_names)]
        relevant_ingredients = set(selected)

    # Crea il database filtrato
    ingredient_db_string = "\n".join([
        f"- {name}: CHO={ingredient_data[name].cho_per_100g}/100g, Vegan={ingredient_data[name].is_vegan}, Veg={ingredient_data[name].is_vegetarian}, GF={ingredient_data[name].is_gluten_free}, LF={ingredient_data[name].is_lactose_free}"
        for name in relevant_ingredients if name in ingredient_data
    ])

    # Implementa un semplice meccanismo di retry con backoff
    max_retries = 2
    retry_delay = 1  # secondi

    for attempt in range(max_retries + 1):
        try:
            # Esegui la chain per questa ricetta specifica
            response_str = modifier_chain.invoke({
                "recipe_name": recipe_name,
                "original_ingredients": original_ingredients_str,
                "target_cho": preferences.target_cho,
                "cho_tolerance": cho_tolerance,
                "vegan": preferences.vegan,
                "vegetarian": preferences.vegetarian,
                "gluten_free": preferences.gluten_free,
                "lactose_free": preferences.lactose_free,
                "ingredient_database_info": ingredient_db_string,
            })

            # Estrai e valida il JSON
            try:
                llm_output = extract_json_from_llm_response(response_str)
                # Gestisci il caso in cui sia restituito solo un errore
                if "error" in llm_output and len(llm_output) == 1:
                    print(
                        f"Thread: Errore dall'LLM per '{recipe_name}': {llm_output['error']}")
                    return None  # Salta questa ricetta

                # Valida con Pydantic
                validated_output = ModifiedRecipeOutput.model_validate(
                    llm_output)

                if validated_output.error:
                    print(
                        f"Thread: Errore dall'LLM per '{recipe_name}': {validated_output.error}")
                    return None  # Salta questa ricetta

                # Ricostruisci la lista di ingredienti con il contributo CHO calcolato
                final_ingredients_list = [
                    RecipeIngredient(
                        name=ing["name"], quantity_g=float(ing["quantity_g"]))
                    for ing in validated_output.final_ingredients
                ]

                # Ricalcola il contributo CHO per ogni ingrediente
                calculated_ingredients_details = calculate_ingredient_cho_contribution(
                    final_ingredients_list, ingredient_data)
                recalculated_total_cho = round(
                    sum(ing.cho_contribution for ing in calculated_ingredients_details), 2)

                # Verifica discrepanza CHO
                if abs(recalculated_total_cho - validated_output.final_total_cho) > 1.0:
                    print(
                        f"Thread: Attenzione: CHO ricalcolato ({recalculated_total_cho}g) differisce da CHO LLM ({validated_output.final_total_cho}g) per '{validated_output.recipe_name}'.")

                final_recipe = FinalRecipeOption(
                    name=validated_output.recipe_name,
                    ingredients=calculated_ingredients_details,
                    total_cho=recalculated_total_cho,  # Usa sempre il valore ricalcolato
                    is_vegan=validated_output.is_vegan,
                    is_vegetarian=validated_output.is_vegetarian,
                    is_gluten_free=validated_output.is_gluten_free,
                    is_lactose_free=validated_output.is_lactose_free
                )

                print(
                    f"Thread: Ricetta '{final_recipe.name}' processata con successo.")
                return final_recipe

            except (json.JSONDecodeError, ValidationError, KeyError, ValueError) as json_error:
                if attempt < max_retries:
                    print(
                        f"Thread: Errore parsing/validazione JSON per ricetta '{recipe_name}'. Retry {attempt+1}/{max_retries}")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                print(
                    f"Thread: Errore parsing/validazione JSON per ricetta '{recipe_name}': {json_error}")
                return None

        except Exception as e:
            if attempt < max_retries:
                print(
                    f"Thread: Errore API per ricetta '{recipe_name}'. Retry {attempt+1}/{max_retries}")
                time.sleep(retry_delay * (attempt + 1))
                continue
            print(f"Thread: Errore API per ricetta '{recipe_name}': {e}")
            return None


# --- Funzione Agente/Nodo ---
def recipe_modifier_agent(state: GraphState) -> GraphState:
    """
    Node Function: Analizza e modifica le ricette usando un LLM, ora con parallelizzazione.
    """
    print("--- ESECUZIONE NODO: Analisi e Modifica Ricette ---")
    preferences: UserPreferences = state['user_preferences']
    initial_recipes: List[Recipe] = state['initial_recipes']
    ingredient_data: Dict[str, IngredientInfo] = state['available_ingredients']

    # Recupera la chiave API di OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        state['error_message'] = "API Key OpenAI non trovata. Assicurati sia nel file .env"
        print("Errore: Chiave API OpenAI non trovata.")
        state['processed_recipes'] = []
        return state

    # Pre-filtering delle ricette per ridurre le chiamate API necessarie
    cho_tolerance = 6.0  # Tolleranza standard
    if len(initial_recipes) > 15:
        print(
            f"Pre-filtering di {len(initial_recipes)} ricette in base al target CHO...")
        filtered_recipes = filter_by_initial_cho(
            initial_recipes, ingredient_data, preferences.target_cho, tolerance=15.0)
        print(
            f"Filtrate {len(filtered_recipes)} ricette per elaborazione prioritaria (su {len(initial_recipes)} totali)")
    else:
        filtered_recipes = initial_recipes

    # Inizializza il modello LLM

    model_name = "gpt-3.5-turbo"
    print(f"Utilizzo modello {model_name} per l'elaborazione delle ricette")

    llm = ChatOpenAI(temperature=0.2, model_name=model_name,
                     openai_api_key=api_key)

    # Prepara il prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", MODIFIER_SYSTEM_PROMPT),
        ("human", """
Analizza e, se necessario, modifica la seguente ricetta per soddisfare i criteri:

**Ricetta Originale:**
Nome: {recipe_name}
Ingredienti Originali: {original_ingredients}

**Criteri da Soddisfare:**
Target CHO: {target_cho} grammi
Tolleranza CHO: +/- {cho_tolerance} grammi
Preferenze Dietetiche: Vegano={vegan}, Vegetariano={vegetarian}, Senza Glutine={gluten_free}, Senza Lattosio={lactose_free}

**Database Ingredienti Disponibili (nome, cho/100g, flags dietetici):**
{ingredient_database_info}

Ricorda di seguire esattamente il formato JSON specificato nel system prompt.
""")
    ])

    # Crea la chain Langchain
    modifier_chain = prompt | llm | StrOutputParser()

    # Impostazioni per l'elaborazione
    # Un numero elevato potrebbe causare limiti di rate per l'API OpenAI
    # Aumentato a 15 worker paralleli per performance
    max_workers = min(15, len(filtered_recipes))

    print(
        f"Avvio modifica per {len(filtered_recipes)} ricette con {max_workers} worker paralleli...")

    # Elaborazione parallela
    processed_recipes = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepara i futures per tutte le ricette
        future_to_recipe = {
            executor.submit(
                process_single_recipe,
                recipe,
                preferences,
                ingredient_data,
                modifier_chain,
                cho_tolerance
            ): recipe.name for recipe in filtered_recipes
        }

        # Raccolta dei risultati man mano che completano
        for future in future_to_recipe:
            recipe_name = future_to_recipe[future]
            try:
                result = future.result()
                if result:  # Se abbiamo un risultato valido (non None)
                    processed_recipes.append(result)
                    print(
                        f"Ricetta '{recipe_name}' aggiunta alla lista processata.")
            except Exception as exc:
                print(f"Ricetta '{recipe_name}' generò un'eccezione: {exc}")

    print(
        f"--- Modifica completata. Ricette processate: {len(processed_recipes)} ---")

    # Aggiorna lo stato
    state['processed_recipes'] = processed_recipes

    # Gestione errori
    if not processed_recipes:
        state['error_message'] = "Nessuna ricetta è stata modificata con successo."
    else:
        # Rimuovi errori precedenti se almeno una ricetta è stata elaborata con successo
        state.pop('error_message', None)

    return state
