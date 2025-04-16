# agents/generator_agent.py

import json
import os
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field as PydanticField, ValidationError
from pydantic.config import ConfigDict

from model_schema import IngredientInfo, FinalRecipeOption, RecipeIngredient, UserPreferences, GraphState
from utils import calculate_ingredient_cho_contribution


class GeneratedRecipeOutput(BaseModel):
    """Struttura Pydantic per l'output JSON atteso dall'LLM Generator."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    recipe_name: str = PydanticField(description="Nome della ricetta generata")
    description: Optional[str] = PydanticField(
        None, description="Breve descrizione della ricetta")
    ingredients: List[Dict[str, Any]] = PydanticField(
        description="Lista di ingredienti [{'name': str, 'quantity_g': float}]")
    is_vegan: bool = PydanticField(
        description="Flag vegano per la ricetta")
    is_vegetarian: bool = PydanticField(
        description="Flag vegetariano per la ricetta")
    is_gluten_free: bool = PydanticField(
        description="Flag senza glutine per la ricetta")
    is_lactose_free: bool = PydanticField(
        description="Flag senza lattosio per la ricetta")
    instructions: Optional[List[str]] = PydanticField(
        None, description="Lista di istruzioni per preparare la ricetta")
    error: Optional[str] = PydanticField(
        None, description="Messaggio di errore se la generazione fallisce")


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


def generate_single_recipe(preferences: UserPreferences, ingredient_data: Dict[str, IngredientInfo],
                           generator_chain, recipe_index: int) -> Optional[FinalRecipeOption]:
    """
    Genera una singola ricetta usando l'LLM e restituisce la ricetta come FinalRecipeOption.
    """
    print(f"Thread: Generazione ricetta #{recipe_index+1}")

    # Costruisci la stringa delle preferenze dietetiche per il prompt
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

    # Prepara la lista degli ingredienti per il prompt
    # Include solo ingredienti che rispettano le preferenze dietetiche
    valid_ingredients = []
    relevant_ingredients = []

    for name, info in ingredient_data.items():
        # Filtro basato su preferenze dietetiche
        if (preferences.vegan and not info.is_vegan) or \
           (preferences.vegetarian and not info.is_vegetarian) or \
           (preferences.gluten_free and not info.is_gluten_free) or \
           (preferences.lactose_free and not info.is_lactose_free):
            continue

        # Aggiungi questo ingrediente alla lista degli ingredienti validi
        valid_ingredients.append(name)

        # Includi solo ingredienti con CHO definito
        if info.cho_per_100g is not None:
            # Crea descrizione dell'ingrediente con informazioni nutrizionali disponibili
            ing_desc = f"{name} (CHO: {info.cho_per_100g}g per 100g"

            if hasattr(info, 'calories_per_100g') and info.calories_per_100g is not None:
                ing_desc += f", Calorie: {info.calories_per_100g} kcal"
            if hasattr(info, 'protein_per_100g') and info.protein_per_100g is not None:
                ing_desc += f", Proteine: {info.protein_per_100g}g"
            if hasattr(info, 'fat_per_100g') and info.fat_per_100g is not None:
                ing_desc += f", Grassi: {info.fat_per_100g}g"
            if hasattr(info, 'fiber_per_100g') and info.fiber_per_100g is not None:
                ing_desc += f", Fibre: {info.fiber_per_100g}g"
            if hasattr(info, 'food_group') and info.food_group is not None:
                ing_desc += f", Gruppo: {info.food_group}"

            ing_desc += ")"
            relevant_ingredients.append(ing_desc)

    # Crea una lista semplice di nomi di ingredienti validi
    valid_ingredients_list = ", ".join(valid_ingredients)

    # Limita il numero di ingredienti per non superare i limiti del prompt
    max_ingredients = 100
    if len(relevant_ingredients) > max_ingredients:
        # Campiona in modo casuale, ma assicurati di avere una varietà di gruppi alimentari
        import random
        # Per risultati deterministici ma diversi per ogni ricetta
        random.seed(recipe_index)
        relevant_ingredients = random.sample(
            relevant_ingredients, max_ingredients)

    # Prepara la lista di esempi di ingredienti ad alto contenuto di CHO
    high_cho_ingredients = []
    medium_cho_ingredients = []

    for name, info in ingredient_data.items():
        if not (preferences.vegan and not info.is_vegan) and \
           not (preferences.vegetarian and not info.is_vegetarian) and \
           not (preferences.gluten_free and not info.is_gluten_free) and \
           not (preferences.lactose_free and not info.is_lactose_free):

            if info.cho_per_100g is not None:
                if info.cho_per_100g > 50:  # Ingredienti ad alto contenuto di CHO
                    high_cho_ingredients.append(
                        f"{name} ({info.cho_per_100g}g CHO per 100g)")
                elif info.cho_per_100g > 20:  # Ingredienti a medio contenuto di CHO
                    medium_cho_ingredients.append(
                        f"{name} ({info.cho_per_100g}g CHO per 100g)")

    # Limita gli esempi a 10 per ciascuna categoria
    if len(high_cho_ingredients) > 10:
        high_cho_ingredients = high_cho_ingredients[:10]
    if len(medium_cho_ingredients) > 10:
        medium_cho_ingredients = medium_cho_ingredients[:10]

    high_cho_examples = ", ".join(high_cho_ingredients)
    medium_cho_examples = ", ".join(medium_cho_ingredients)

    ingredients_list_string = "\n".join(
        [f"- {ing}" for ing in relevant_ingredients])

    # Implementa un semplice meccanismo di retry con backoff
    max_retries = 2
    retry_delay = 1  # secondi

    for attempt in range(max_retries + 1):
        try:
            # Esegui la chain per generare la ricetta
            response_str = generator_chain.invoke({
                "target_cho": preferences.target_cho,
                "recipe_index": recipe_index + 1,
                "dietary_preferences": dietary_preferences_string,
                "ingredients_list": ingredients_list_string,
                "high_cho_examples": high_cho_examples,
                "medium_cho_examples": medium_cho_examples,
                "valid_ingredients": valid_ingredients_list
            })

            # Estrai e valida il JSON
            try:
                llm_output = extract_json_from_llm_response(response_str)
                # Gestisci il caso in cui sia restituito solo un errore
                if "error" in llm_output and len(llm_output) == 1:
                    print(
                        f"Thread: Errore dall'LLM per ricetta #{recipe_index+1}: {llm_output['error']}")
                    return None  # Salta questa ricetta

                # Valida con Pydantic
                validated_output = GeneratedRecipeOutput.model_validate(
                    llm_output)

                if validated_output.error:
                    print(
                        f"Thread: Errore dall'LLM per ricetta #{recipe_index+1}: {validated_output.error}")
                    return None  # Salta questa ricetta

                # Verifica che tutti gli ingredienti siano nella lista degli ingredienti validi
                invalid_ingredients = []
                for ing in validated_output.ingredients:
                    if ing["name"] not in valid_ingredients:
                        invalid_ingredients.append(ing["name"])

                if invalid_ingredients:
                    print(
                        f"Thread: Ricetta #{recipe_index+1} contiene ingredienti non validi: {', '.join(invalid_ingredients)}. Retry.")
                    if attempt < max_retries:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        print(
                            "Thread: Troppi tentativi con ingredienti non validi. Ricetta scartata.")
                        return None

                # Ricostruisci la lista di ingredienti come RecipeIngredient
                recipe_ingredients = [
                    RecipeIngredient(
                        name=ing["name"],
                        quantity_g=float(ing["quantity_g"])
                    )
                    for ing in validated_output.ingredients
                ]

                # Calcola il contributo CHO e altre informazioni nutrizionali per ogni ingrediente
                calculated_ingredients = calculate_ingredient_cho_contribution(
                    recipe_ingredients, ingredient_data
                )

                # Calcola i totali nutrizionali
                total_cho = round(
                    sum(ing.cho_contribution for ing in calculated_ingredients), 2)

                # Costruisci la ricetta finale
                final_recipe = FinalRecipeOption(
                    name=validated_output.recipe_name,
                    description=validated_output.description,
                    ingredients=calculated_ingredients,
                    total_cho=total_cho,
                    is_vegan=validated_output.is_vegan,
                    is_vegetarian=validated_output.is_vegetarian,
                    is_gluten_free=validated_output.is_gluten_free,
                    is_lactose_free=validated_output.is_lactose_free,
                    instructions=validated_output.instructions
                )

                print(
                    f"Thread: Ricetta #{recipe_index+1} '{final_recipe.name}' generata con successo (CHO: {total_cho}g).")
                return final_recipe

            except (json.JSONDecodeError, ValidationError, KeyError, ValueError) as json_error:
                if attempt < max_retries:
                    print(
                        f"Thread: Errore parsing/validazione JSON per ricetta #{recipe_index+1}. Retry {attempt+1}/{max_retries}")
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                print(
                    f"Thread: Errore parsing/validazione JSON per ricetta #{recipe_index+1}: {json_error}")
                return None

        except Exception as e:
            if attempt < max_retries:
                print(
                    f"Thread: Errore API per ricetta #{recipe_index+1}. Retry {attempt+1}/{max_retries}")
                time.sleep(retry_delay * (attempt + 1))
                continue
            print(f"Thread: Errore API per ricetta #{recipe_index+1}: {e}")
            return None


def generate_recipes_agent(state: GraphState) -> GraphState:
    """
    Node Function: Genera più ricette da zero basate sulle preferenze dell'utente.
    """
    print("--- ESECUZIONE NODO: Generazione Ricette ---")
    preferences: UserPreferences = state['user_preferences']
    ingredient_data: Dict[str, IngredientInfo] = state['available_ingredients']

    # Configura il numero di ricette da generare (più di quelle necessarie per compensare possibili fallimenti)
    # Miriamo a generare 8 ricette per averne poi almeno 3 valide dopo la verifica
    target_recipes = 8  # Aumentiamo da 6 a 8 per avere più possibilità

    # Recupera la chiave API di OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        state['error_message'] = "API Key OpenAI non trovata. Assicurati sia nel file .env"
        print("Errore: Chiave API OpenAI non trovata.")
        state['generated_recipes'] = []
        return state

    # Inizializza il modello LLM
    model_name = "gpt-3.5-turbo"  # O un altro modello adatto
    print(f"Utilizzo modello {model_name} per la generazione delle ricette")
    # Aumenta la temperatura per più creatività e diversità
    llm = ChatOpenAI(temperature=0.8, model_name=model_name,
                     openai_api_key=api_key)

    # PROMPT MIGLIORATO con enfasi sul range CHO, distribuzione bilanciata e diversità
    system_prompt = """
Sei un esperto chef e nutrizionista specializzato nella creazione di ricette bilanciate e personalizzate. Il tuo compito è creare ricette originali che soddisfino precise esigenze nutrizionali e dietetiche.

ISTRUZIONI IMPORTANTI:
- CRUCIALE: Devi creare una ricetta con un target di carboidrati (CHO) di ESATTAMENTE il valore specificato ±5g.
- La ricetta deve avere una distribuzione BILANCIATA di CHO tra gli ingredienti - non più del 70% dei CHO totali deve provenire da un singolo ingrediente.
- Usa SOLO ingredienti dalla lista fornita. Non inventare o aggiungere ingredienti che non sono in questa lista.
- ATTENZIONE: Puoi usare ESCLUSIVAMENTE gli ingredienti che ti fornirò alla fine del prompt. Questi sono gli unici ingredienti disponibili come se fossero gli unici nella dispensa.
- Specifica le quantità ESATTE in grammi per ogni ingrediente.
- Segui rigorosamente le preferenze dietetiche indicate.
- IMPORTANTE: Questa è la ricetta #{recipe_index}. Crea una ricetta COMPLETAMENTE DIVERSA dalle precedenti. Il nome, il tipo di piatto, gli ingredienti principali e lo stile di cucina DEVONO essere diversi da qualsiasi altra ricetta nella sessione.

STRATEGIA PER RAGGIUNGERE IL TARGET CHO:
1. Seleziona una combinazione di ingredienti ad alto, medio e basso contenuto di CHO.
2. Calcola attentamente il contributo in CHO di ogni ingrediente: (quantità_g * CHO_per_100g) / 100
3. Assicurati che il totale sia VICINO al target (entro ±5g).
4. IMPORTANTE: Distribuisci i CHO tra più ingredienti per una ricetta bilanciata.

ESEMPI DI INGREDIENTI E LORO CONTENUTO CHO:
- Ingredienti ad alto contenuto di CHO: {high_cho_examples}
- Ingredienti a medio contenuto di CHO: {medium_cho_examples}

RICETTE BILANCIATE:
- Usa almeno 4-5 ingredienti per un piatto completo.
- Includi una proteina, un carboidrato, verdure/frutta, e grassi sani.
- Quantità ragionevoli: 70-120g di cereali (pasta, riso), 100-200g di proteine, 100-200g di verdure.
- Un singolo ingrediente NON dovrebbe contribuire più del 70% dei CHO totali.

DIVERSITÀ DELLE RICETTE:
- Se stai creando la ricetta #1: scegli liberamente un tipo di piatto.
- Se stai creando la ricetta #2: scegli un tipo di piatto COMPLETAMENTE DIVERSO dalla ricetta #1 (es. se #1 era un primo, fai un secondo o un piatto unico).
- Se stai creando la ricetta #3 o successive: scegli un tipo di piatto diverso dalle precedenti, con cucina di origine diversa (es. mediterranea, asiatica, sudamericana).
- NON ripetere gli stessi ingredienti principali delle ricette precedenti.
- VARIA le tecniche di cottura tra le diverse ricette (cottura al forno, saltato in padella, bollitura, ecc).

FORMATO JSON RICHIESTO:
```json
{{
  "recipe_name": "Nome Creativo della Ricetta",
  "description": "Breve descrizione del piatto e dei suoi sapori",
  "ingredients": [
    {{"name": "Nome Ingrediente 1", "quantity_g": 100}},
    {{"name": "Nome Ingrediente 2", "quantity_g": 50}}
  ],
  "is_vegan": true,
  "is_vegetarian": true,
  "is_gluten_free": false,
  "is_lactose_free": true,
  "instructions": [
    "Passo 1: Descrizione dettagliata",
    "Passo 2: Descrizione dettagliata"
  ]
}}
```

ASSICURATI CHE:
- Ogni ingrediente DEVE esistere ESATTAMENTE nella lista di ingredienti validi.
- Il valore totale di CHO DEVE essere entro ±5g del target specificato.
- I CHO siano DISTRIBUITI tra più ingredienti (max 70% da un singolo ingrediente).
- Le quantità siano numeri realistici in grammi.
- I valori booleani riflettano accuratamente le proprietà della ricetta.
- Le istruzioni siano chiare e complete.
- Il nome e il tipo di piatto deve essere DIVERSO dalle ricette precedenti.
"""

    human_prompt = """
Genera la ricetta #{recipe_index} rispettando questi criteri:

TARGET NUTRIZIONALE:
- Carboidrati (CHO): {target_cho}g (±5g) - QUESTO È CRUCIALE!
- Distribuzione: Nessun ingrediente deve fornire più del 70% dei CHO totali.

PREFERENZE DIETETICHE:
{dietary_preferences}

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
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    # Crea la chain Langchain
    generator_chain = prompt | llm | StrOutputParser()

    # Configurazione per generazione parallela
    # Limita a 3 worker per non superare rate limit API
    max_workers = min(3, target_recipes)

    print(
        f"Avvio generazione di {target_recipes} ricette con {max_workers} worker paralleli...")

    # Generazione parallela delle ricette
    generated_recipes = []
    recipe_names = set()  # Mantiene traccia dei nomi delle ricette già generate

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepara i futures per tutte le ricette da generare
        futures = [
            executor.submit(
                generate_single_recipe,
                preferences,
                ingredient_data,
                generator_chain,
                i
            ) for i in range(target_recipes)
        ]

        # Raccolta dei risultati man mano che completano
        for future in futures:
            try:
                result = future.result()
                if result:  # Se abbiamo un risultato valido (non None)
                    # Verifica che la ricetta sia unica controllando il nome
                    if result.name not in recipe_names:
                        recipe_names.add(result.name)
                        generated_recipes.append(result)
                        print(
                            f"Ricetta '{result.name}' aggiunta alla lista di ricette generate.")
                    else:
                        print(
                            f"Ricetta '{result.name}' scartata perché duplicata.")
            except Exception as exc:
                print(
                    f"La generazione di una ricetta ha generato un'eccezione: {exc}")

    # Se abbiamo generato meno ricette del necessario, tenta di generarne altre
    # con una temperatura più alta per aumentare la diversità
    if len(generated_recipes) < 3:
        # Genera almeno 2 ricette aggiuntive
        remaining_to_generate = max(3 - len(generated_recipes), 2)
        print(
            f"Generazione di {remaining_to_generate} ricette aggiuntive con più diversità...")

        # Usa una temperatura più alta per maggiore creatività
        llm_diverse = ChatOpenAI(temperature=0.95, model_name=model_name,
                                 openai_api_key=api_key)

        # Modifichiamo leggermente il prompt per enfatizzare ancora di più la diversità
        diverse_system_prompt = system_prompt + \
            "\n\nNOTA SPECIALE: È ASSOLUTAMENTE ESSENZIALE che questa ricetta sia COMPLETAMENTE DIVERSA dalle precedenti. Scegli una cucina etnica, un metodo di cottura e ingredienti principali DIFFERENTI."

        diverse_prompt = ChatPromptTemplate.from_messages([
            ("system", diverse_system_prompt),
            ("human", human_prompt)
        ])

        generator_chain_diverse = diverse_prompt | llm_diverse | StrOutputParser()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            additional_futures = [
                executor.submit(
                    generate_single_recipe,
                    preferences,
                    ingredient_data,
                    generator_chain_diverse,
                    i + target_recipes  # Usiamo indici diversi per forzare maggiore varietà
                ) for i in range(remaining_to_generate)
            ]

            for future in additional_futures:
                try:
                    result = future.result()
                    if result and result.name not in recipe_names:  # Verifica unicità
                        recipe_names.add(result.name)
                        generated_recipes.append(result)
                        print(
                            f"Ricetta aggiuntiva '{result.name}' aggiunta alla lista.")
                except Exception as exc:
                    print(
                        f"La generazione di una ricetta aggiuntiva ha generato un'eccezione: {exc}")

    # Se abbiamo ancora dei duplicati, forzalo a verificare la diversità degli ingredienti
    final_recipes = []
    ingredient_sets = []

    for recipe in generated_recipes:
        # Crea un set degli ingredienti principali (escludendo ingredienti di base o condimenti)
        main_ingredients = set([ing.name for ing in recipe.ingredients
                               # Consideriamo solo ingredienti con contributo CHO significativo
                                if ing.cho_contribution > 5.0])

        # Verifica se questa combinazione di ingredienti principali è già stata usata
        is_unique = True
        for existing_ingredients in ingredient_sets:
            # Se più del 60% degli ingredienti principali si sovrappongono, consideriamo la ricetta come simile
            if len(main_ingredients.intersection(existing_ingredients)) / len(main_ingredients) > 0.6:
                is_unique = False
                print(
                    f"Ricetta '{recipe.name}' scartata perché usa ingredienti principali simili ad altra ricetta.")
                break

        if is_unique:
            ingredient_sets.append(main_ingredients)
            final_recipes.append(recipe)

    # Assicuriamoci di mantenere almeno 3 ricette se possibile
    if len(final_recipes) < 3 and len(generated_recipes) >= 3:
        # Se abbiamo scartato troppe ricette, teniamo le prime 3 delle ricette generate originariamente
        print("Mantenimento di almeno 3 ricette, anche se alcune hanno ingredienti simili.")
        final_recipes = generated_recipes[:3]

    print(
        f"--- Generazione completata. Ricette uniche generate: {len(final_recipes)}/{target_recipes} ---")

    # Aggiorna lo stato
    state['generated_recipes'] = final_recipes

    # Gestione errori
    if not final_recipes:
        state['error_message'] = "Nessuna ricetta è stata generata con successo."
    else:
        # Rimuovi errori precedenti se almeno una ricetta è stata generata con successo
        state.pop('error_message', None)

    return state
