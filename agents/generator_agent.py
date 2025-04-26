import json
import random
import os
import time
from concurrent.futures import ThreadPoolExecutor

# Import LLM e componenti Langchain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import modelli Pydantic e schema stato
from pydantic import BaseModel, Field as PydanticField, ValidationError
from pydantic.config import ConfigDict
from model_schema import IngredientInfo, FinalRecipeOption, RecipeIngredient, UserPreferences, GraphState, CalculatedIngredient

# --- Modelli Pydantic per l'output dell'LLM ---


class GeneratedRecipeOutput(BaseModel):
    """
    Modello Pydantic che rappresenta il formato di output atteso da parte dell'LLM.

    Include informazioni su nome, descrizione, ingredienti, caratteristiche dietetiche
    e istruzioni per la preparazione.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    recipe_name: str = PydanticField(description="Nome della ricetta generata")
    description: str = PydanticField(
        description="Breve descrizione", default="")
    ingredients: list[dict] = PydanticField(
        description="Lista ingredienti [{'name': str, 'quantity_g': float}]")
    is_vegan: bool = PydanticField(
        description="Flag vegano LLM", default=False)
    is_vegetarian: bool = PydanticField(
        description="Flag vegetariano LLM", default=False)
    is_gluten_free: bool = PydanticField(
        description="Flag senza glutine LLM", default=False)
    is_lactose_free: bool = PydanticField(
        description="Flag senza lattosio LLM", default=False)
    instructions: list[str] = PydanticField(
        description="Lista istruzioni", default=[])

# --- Funzione estrazione JSON ---


def extract_json_from_llm_response(response_str: str) -> dict:
    """Estrae JSON dalla risposta dell'LLM utilizzando vari metodi."""
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

# --- Funzione Worker per Generare Singola Ricetta ---


def generate_single_recipe(
    preferences: UserPreferences,
    generator_chain: any,
    recipe_index: int
) -> dict:
    """
    Genera una singola ricetta creativa con vincoli minimi.
    Non effettua validazione o matching degli ingredienti.
    """
    print(f"Thread: Generazione ricetta #{recipe_index+1}")

    # Costruisci stringa preferenze
    dietary_preferences = []
    if preferences.vegan:
        dietary_preferences.append("vegana")
    elif preferences.vegetarian:
        dietary_preferences.append("vegetariana")
    if preferences.gluten_free:
        dietary_preferences.append("senza glutine")
    if preferences.lactose_free:
        dietary_preferences.append("senza lattosio")
    dietary_preferences_string = ", ".join(
        dietary_preferences) if dietary_preferences else "nessuna preferenza specifica"

    # Tentativo di generazione con retry minimo
    max_retries = 1
    retry_delay = 1

    for attempt in range(max_retries + 1):
        try:
            # Esegui chain LLM
            response_str = generator_chain.invoke({
                "target_cho": preferences.target_cho,
                "recipe_index": recipe_index + 1,
                "dietary_preferences": dietary_preferences_string
            })

            try:
                # Processa risposta
                llm_output = extract_json_from_llm_response(response_str)
                print(
                    f"Thread: Ricetta #{recipe_index+1} JSON estratto correttamente")

                # Validazione semplice della struttura JSON
                validated_output = GeneratedRecipeOutput.model_validate(
                    llm_output)

                return {
                    "recipe_name": validated_output.recipe_name,
                    "description": validated_output.description,
                    "ingredients": validated_output.ingredients,
                    "is_vegan": validated_output.is_vegan,
                    "is_vegetarian": validated_output.is_vegetarian,
                    "is_gluten_free": validated_output.is_gluten_free,
                    "is_lactose_free": validated_output.is_lactose_free,
                    "instructions": validated_output.instructions
                }

            except (json.JSONDecodeError, ValidationError, ValueError) as json_error:
                if attempt < max_retries:
                    print(
                        f"Thread: Errore JSON ricetta #{recipe_index+1}: {json_error}. Retry.")
                    time.sleep(retry_delay)
                    continue
                print(
                    f"Thread: Errore JSON definitivo ricetta #{recipe_index+1}: {json_error}")
                return None

        except Exception as e:
            if attempt < max_retries:
                print(
                    f"Thread: Errore API ricetta #{recipe_index+1}: {e}. Retry.")
                time.sleep(retry_delay)
                continue
            print(
                f"Thread: Errore API definitivo ricetta #{recipe_index+1}: {e}")
            return None

    return None  # Fallimento dopo tutti i tentativi

# --- Funzione Agente Principale ---


def generate_recipes_agent(state: GraphState) -> GraphState:
    """
    Node Function: Genera ricette creative con vincoli minimi.
    Non effettua validazione o matching degli ingredienti.
    """
    print("--- ESECUZIONE NODO: Generazione Ricette (Semplificata) ---")

    try:
        # Recupera preferenze
        preferences = state['user_preferences']

        target_cho = preferences.target_cho  # Ottieni il valore
        # TRACCIA!
        print(f"Valore target_cho ricevuto dallo stato: {target_cho}")

        # Setup LLM, Prompt
        target_recipes = 10  # Numero ricette da tentare
        max_workers = 8
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            state['error_message'] = "API Key OpenAI non trovata."
            print(f"Errore: {state['error_message']}")
            state['generated_recipes'] = []
            return state

        model_name = "gpt-3.5-turbo"
        print(
            f"Utilizzo modello {model_name} per generazione ricette creative")
        llm = ChatOpenAI(temperature=0.9, model_name=model_name,
                         openai_api_key=api_key)

        # PROMPT SEMPLIFICATO - CORRETTO
        system_prompt = """
        **RUOLO**:  Sei uno chef creativo esperto nella creazione di ricette originali e gustose.

        **COMPITO**: Genera una ricetta PER UNA PERSONA che abbia ESATTAMENTE {target_cho}g di carboidrati totali (questo è ASSOLUTAMENTE CRITICO).

        ---

        ## LINEE GUIDA

        - **Target Carboidrati**(CHO): OGNI ricetta DEVE contenere {target_cho}g di carboidrati totali. Questo è il requisito PIÙ IMPORTANTE. Se non rispetti questo requisito, la ricetta verrà scartata.
        - **Preferenze Dietetiche**: Rispetta le seguenti preferenze: {dietary_preferences}
        - **Ingredienti**: Usa ingredienti comuni e facilmente reperibili.
        - **Quantità**: Specifica le quantità in grammi per ogni ingrediente.
        
        
        ## NON PREOCCUPARTI DI:
        - Calcolare con precisione esatta i carboidrati
        - Bilanciare perfettamente i macronutrienti
        - Rispettare specifici limiti di ingredienti
        
        ## FORMATO OUTPUT

        Fornisci la ricetta nel seguente formato JSON:

        ```json
        {{
        "recipe_name": "Nome Creativo della Ricetta",
        "description": "Breve descrizione accattivante del piatto",
        "ingredients": [
            {{"name": "Nome Ingrediente 1", "quantity_g": 120.0}},
            {{"name": "Nome Ingrediente 2", "quantity_g": 85.5}}
        ],
        "is_vegan": boolean,
        "is_vegetarian": boolean,
        "is_gluten_free": boolean,
        "is_lactose_free": boolean,
        "instructions": [
            "Passo 1: Istruzione chiara.",
            "Passo 2: Altra istruzione."
        ]
        }}
        ```
        """

        human_prompt = """
        Genera la ricetta {recipe_index} che DEVE contenere ESATTAMENTE {target_cho}g di carboidrati totali.
        Ricorda:
        - Il target CHO di {target_cho}g è ASSOLUTAMENTE CRITICO
        - La ricetta deve essere adatta a: {dietary_preferences}
        - Seleziona attentamente gli ingredienti e le loro quantità per raggiungere questo target

        Sii creativo e proponi un piatto originale, gustoso e realizzabile.
        """

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_prompt)])
        generator_chain = prompt | llm | StrOutputParser()

        print(
            f"Avvio generazione di {target_recipes} ricette creative con {max_workers} worker paralleli...")
        raw_recipes = []

        # Esecuzione Parallela (semplificata)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    generate_single_recipe,  # Funzione worker
                    preferences,            # Preferenze
                    generator_chain,        # Chain LLM
                    i                       # Indice ricetta
                ) for i in range(target_recipes)
            ]

            # Raccolta Risultati
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        raw_recipes.append(result)
                        print(f"Ricetta '{result['recipe_name']}' aggiunta.")
                except Exception as exc:
                    print(f"Generazione ricetta fallita con eccezione: {exc}")

        # Conversione in ricette non verificate per il prossimo step
        unverified_recipes = []
        for raw_recipe in raw_recipes:
            try:
                # Converti la struttura raw in CalculatedIngredient direttamente (non RecipeIngredient)
                calculated_ingredients = []
                for ing_data in raw_recipe["ingredients"]:
                    # Gestisci sia "quantity_g" che possibili alternative
                    quantity = None
                    if "quantity_g" in ing_data:
                        quantity = float(ing_data["quantity_g"])
                    elif "quantity" in ing_data:
                        quantity = float(ing_data["quantity"])
                    elif "amount_g" in ing_data:
                        quantity = float(ing_data["amount_g"])
                    else:
                        # Default se non trovata
                        quantity = 0.0

                    # Crea CalculatedIngredient invece di RecipeIngredient
                    calculated_ingredients.append(
                        CalculatedIngredient(
                            name=ing_data["name"],
                            quantity_g=quantity,
                            cho_per_100g=None,
                            cho_contribution=None,
                            original_llm_name=ing_data["name"]
                        )
                    )

                # Crea FinalRecipeOption con CalculatedIngredient
                recipe = FinalRecipeOption(
                    name=raw_recipe["recipe_name"],
                    description=raw_recipe.get("description", ""),
                    # Ora sono CalculatedIngredient, non RecipeIngredient
                    ingredients=calculated_ingredients,
                    total_cho=None,  # Da calcolare nel verificatore
                    total_calories=None,
                    total_protein_g=None,
                    total_fat_g=None,
                    total_fiber_g=None,
                    is_vegan=raw_recipe.get("is_vegan", False),
                    is_vegetarian=raw_recipe.get("is_vegetarian", False),
                    is_gluten_free=raw_recipe.get("is_gluten_free", False),
                    is_lactose_free=raw_recipe.get("is_lactose_free", False),
                    instructions=raw_recipe.get("instructions", [])
                )
                unverified_recipes.append(recipe)
                print(
                    f"Conversione riuscita per ricetta '{recipe.name}' con {len(calculated_ingredients)} ingredienti")
            except Exception as e:
                print(
                    f"Errore nella conversione della ricetta '{raw_recipe.get('recipe_name', 'sconosciuta')}': {e}")

        print(
            f"--- Generazione completata. Ricette raw generate: {len(unverified_recipes)} ---")
        state['generated_recipes'] = unverified_recipes

        if not unverified_recipes:
            if not state.get('error_message'):
                state['error_message'] = "Nessuna ricetta valida generata. Il sistema non è riuscito a creare ricette con la struttura corretta."
        else:
            state.pop('error_message', None)

        return state

    except Exception as ex:
        print(f"Errore imprevisto durante la generazione: {ex}")
        import traceback
        traceback.print_exc()
        state['error_message'] = f"Errore imprevisto durante la generazione ricette: {ex}"
        state['generated_recipes'] = []
        return state
