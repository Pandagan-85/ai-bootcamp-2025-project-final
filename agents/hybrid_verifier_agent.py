import os
from langchain_openai import ChatOpenAI
from model_schema import GraphState, FinalRecipeOption, RecipeIngredient
from langchain_core.prompts import ChatPromptTemplate
import json
import re
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
from copy import deepcopy
from utils import calculate_ingredient_cho_contribution


def adjust_recipe_cho(recipe: FinalRecipeOption, target_cho: float, ingredient_data: Dict) -> FinalRecipeOption:
    """
    Aggiusta le quantità degli ingredienti per raggiungere il target CHO desiderato.
    Versione aggressiva che aggiusta anche ricette con CHO molto lontani dal target.

    Args:
        recipe: La ricetta da modificare
        target_cho: Il valore CHO target desiderato
        ingredient_data: Dizionario con informazioni nutrizionali degli ingredienti

    Returns:
        La ricetta modificata con quantità aggiustate
    """
    # Se siamo già nel target, non fare nulla
    if abs(recipe.total_cho - target_cho) < 3.0:
        return recipe

    # Crea una copia profonda della ricetta per non modificare l'originale
    adjusted_recipe = deepcopy(recipe)

    # Identifica gli ingredienti ricchi di CHO (più del 5% del totale)
    cho_rich_ingredients = []
    for ing in adjusted_recipe.ingredients:
        if ing.cho_contribution > 0:
            # Se la ricetta ha pochi CHO totali, consideriamo tutti gli ingredienti con CHO
            if adjusted_recipe.total_cho < 30 or (ing.cho_contribution / adjusted_recipe.total_cho) > 0.05:
                cho_rich_ingredients.append(ing)

    # Se non ci sono ingredienti ricchi di CHO, verifica se ci sono nel database
    if not cho_rich_ingredients:
        # Verifica se ci sono altri ingredienti con CHO nel database che possiamo aumentare
        potential_ingredients = []
        for ing in adjusted_recipe.ingredients:
            if ing.name in ingredient_data and ingredient_data[ing.name].cho_per_100g > 5:
                potential_ingredients.append(ing)

        # Se troviamo ingredienti potenziali, usiamo quelli
        if potential_ingredients:
            cho_rich_ingredients = potential_ingredients

    # Se ancora non ci sono ingredienti ricchi di CHO, non possiamo modificare
    if not cho_rich_ingredients:
        return recipe

    # Ordina per contributo CHO decrescente
    cho_rich_ingredients.sort(key=lambda x: x.cho_contribution, reverse=True)

    # Calcola di quanto dobbiamo modificare i CHO
    cho_diff = target_cho - adjusted_recipe.total_cho

    # Calcola il fattore di scala necessario per tutta la ricetta
    if adjusted_recipe.total_cho > 0:
        # Se la ricetta ha un valore CHO significativo, possiamo semplicamente scalare tutto
        scaling_factor = target_cho / adjusted_recipe.total_cho

        # Limiti più permissivi per le ricette lontane dal target
        if abs(cho_diff) > 20:
            # Per casi estremi, consentiamo modifiche più drastiche
            min_scale = 0.5 if adjusted_recipe.total_cho > target_cho else 1.0
            max_scale = 1.0 if adjusted_recipe.total_cho > target_cho else 3.0
        else:
            # Per casi normali, limiti standard
            min_scale = 0.6
            max_scale = 2.0

        # Limita il fattore di scala entro i limiti
        scaling_factor = max(min_scale, min(scaling_factor, max_scale))

        # Applica il fattore di scala a tutti gli ingredienti ricchi di CHO
        recipe_ingredients = []
        for ing in adjusted_recipe.ingredients:
            # Applica scaling solo agli ingredienti ricchi di CHO
            if ing in cho_rich_ingredients:
                recipe_ingredients.append(
                    RecipeIngredient(
                        name=ing.name,
                        quantity_g=ing.quantity_g * scaling_factor
                    )
                )
            else:
                recipe_ingredients.append(
                    RecipeIngredient(
                        name=ing.name,
                        quantity_g=ing.quantity_g
                    )
                )

        # Ricalcola i contributi nutrizionali
        updated_ingredients = calculate_ingredient_cho_contribution(
            recipe_ingredients, ingredient_data)
        adjusted_recipe.ingredients = updated_ingredients
        adjusted_recipe.total_cho = round(
            sum(ing.cho_contribution for ing in updated_ingredients), 2)

    # Se necessario, fai un aggiustamento fine sull'ingrediente principale
    if abs(adjusted_recipe.total_cho - target_cho) > 5.0 and cho_rich_ingredients:
        main_ingredient = cho_rich_ingredients[0]
        ing_name = main_ingredient.name

        if ing_name in ingredient_data and ingredient_data[ing_name].cho_per_100g > 0:
            cho_per_100g = ingredient_data[ing_name].cho_per_100g

            # Ricalcola la differenza dopo il primo aggiustamento
            cho_diff = target_cho - adjusted_recipe.total_cho

            # Calcola la modifica necessaria
            gram_diff = (cho_diff / cho_per_100g) * 100.0

            # Trova l'ingrediente da modificare
            for i, ing in enumerate(adjusted_recipe.ingredients):
                if ing.name == ing_name:
                    # Calcola la nuova quantità
                    new_quantity = ing.quantity_g + gram_diff

                    # Assicurati che rimanga ragionevole (minimo 10g)
                    new_quantity = max(10.0, new_quantity)

                    # Aggiorna l'ingrediente
                    recipe_ingredients = [
                        RecipeIngredient(
                            name=ing.name,
                            quantity_g=new_quantity if ing.name == ing_name else ing.quantity_g
                        ) for ing in adjusted_recipe.ingredients
                    ]

                    # Ricalcola i contributi
                    updated_ingredients = calculate_ingredient_cho_contribution(
                        recipe_ingredients, ingredient_data)
                    adjusted_recipe.ingredients = updated_ingredients
                    adjusted_recipe.total_cho = round(
                        sum(ing.cho_contribution for ing in updated_ingredients), 2)
                    break

    # Aggiungi un'indicazione al nome che è stata modificata
    if recipe.total_cho != adjusted_recipe.total_cho:
        if "Aggiustata" not in recipe.name and "Bilanciata" not in recipe.name:
            adjusted_recipe.name = f"{adjusted_recipe.name} (Aggiustata)"

        # Se hai una descrizione, aggiorna anche quella
        if adjusted_recipe.description and "aggiustate" not in adjusted_recipe.description.lower():
            adjusted_recipe.description = f"{adjusted_recipe.description} Ricetta con quantità aggiustate per raggiungere il target CHO."

    return adjusted_recipe


def validate_recipe_ingredients(recipe: FinalRecipeOption, ingredient_data: Dict) -> bool:
    """
    Verifica che tutti gli ingredienti nella ricetta esistano nel database degli ingredienti.

    Args:
        recipe: La ricetta da verificare
        ingredient_data: Dizionario con informazioni sugli ingredienti

    Returns:
        True se tutti gli ingredienti sono validi, False altrimenti
    """
    invalid_ingredients = []

    for ing in recipe.ingredients:
        if ing.name not in ingredient_data:
            invalid_ingredients.append(ing.name)

    if invalid_ingredients:
        print(
            f"Ricetta '{recipe.name}' contiene ingredienti non validi: {', '.join(invalid_ingredients)}")
        return False

    return True


def hybrid_verifier_agent(state: GraphState) -> GraphState:
    """
    Node Function: Verifica le ricette generate utilizzando una combinazione 
    di regole precise e LLM per la selezione finale.
    """
    print("--- ESECUZIONE NODO: Verifica Ibrida delle Ricette ---")
    preferences = state['user_preferences']
    generated_recipes = state.get('generated_recipes', [])
    ingredient_data = state['available_ingredients']
    target_cho = preferences.target_cho
    cho_tolerance = 10.0
    exact_recipes_required = 3

    if not generated_recipes:
        print("Nessuna ricetta generata da verificare.")
        state['final_verified_recipes'] = []
        if not state.get('error_message'):
            state['error_message'] = "Nessuna ricetta è stata generata con successo."
        return state

    # ------ PARTE 1: VERIFICA CON REGOLE PRECISE ------

    # Filtra le ricette per ingredienti validi
    filtered_recipes = []
    for recipe in generated_recipes:
        if validate_recipe_ingredients(recipe, ingredient_data):
            filtered_recipes.append(recipe)

    # Aggiusta le ricette per CHO
    adjusted_recipes = []
    for recipe in filtered_recipes:
        # Applica aggiustamenti solo se necessario e possibile
        if abs(recipe.total_cho - target_cho) > cho_tolerance:
            adjusted_recipe = adjust_recipe_cho(
                recipe, target_cho, ingredient_data)
            adjusted_recipes.append(adjusted_recipe)
        else:
            adjusted_recipes.append(recipe)

    # Verifica criteri base
    valid_recipes = []
    for recipe in adjusted_recipes:
        # Verifica CHO
        cho_ok = (
            target_cho - cho_tolerance) <= recipe.total_cho <= (target_cho + cho_tolerance)
        if not cho_ok:
            continue

        # Verifica preferenze dietetiche (SOLO quelle richieste dall'utente)
        diet_ok = True
        if preferences.vegan and not recipe.is_vegan:
            diet_ok = False
        if preferences.vegetarian and not recipe.is_vegetarian:
            diet_ok = False
        if preferences.gluten_free and not recipe.is_gluten_free:
            diet_ok = False
        if preferences.lactose_free and not recipe.is_lactose_free:
            diet_ok = False

        if not diet_ok:
            continue

        # Se tutti i controlli base sono OK, aggiungi alla lista
        valid_recipes.append(recipe)

    # Se abbiamo meno di 3 ricette valide, potremmo essere più permissivi
    if len(valid_recipes) < exact_recipes_required:
        print(
            f"Solo {len(valid_recipes)} ricette hanno passato i controlli di base.")
        # Opzionale: logica più permissiva qui

    # ------ PARTE 2: SELEZIONE FINALE CON LLM ------

    # Se abbiamo più di 3 ricette valide, usiamo l'LLM per la selezione finale
    if len(valid_recipes) > exact_recipes_required:
        verified_recipes = select_diverse_recipes_with_llm(
            valid_recipes, preferences)
    else:
        verified_recipes = valid_recipes

    state['final_verified_recipes'] = verified_recipes
    return state


def select_diverse_recipes_with_llm(recipes, preferences):
    """
    Utilizza l'LLM per selezionare le ricette più diverse tra loro.
    """
    # Prepara i dati per il prompt
    recipe_descriptions = []
    for i, recipe in enumerate(recipes):
        # Crea una rappresentazione testuale di ogni ricetta
        ingredients_text = ", ".join(
            [f"{ing.name} ({ing.quantity_g}g)" for ing in recipe.ingredients])
        main_ingredient = max(
            recipe.ingredients, key=lambda ing: ing.cho_contribution).name if recipe.ingredients else "N/A"

        description = f"""
Ricetta #{i+1}: "{recipe.name}"
CHO totali: {recipe.total_cho}g
Ingrediente principale: {main_ingredient}
Tutti gli ingredienti: {ingredients_text}
Categorie: {"Vegana " if recipe.is_vegan else ""}{"Vegetariana " if recipe.is_vegetarian else ""}{"Senza glutine " if recipe.is_gluten_free else ""}{"Senza lattosio" if recipe.is_lactose_free else ""}
        """
        recipe_descriptions.append(description)

    all_recipes_text = "\n\n".join(recipe_descriptions)

    # Costruisci il prompt per l'LLM
    system_prompt = """Sei un esperto chef e nutrizionista. Il tuo compito è selezionare le 3 ricette più diverse tra loro da un elenco, considerando:
1. Tipo di piatto (primo, secondo, contorno, ecc.)
2. Ingredienti principali utilizzati
3. Stile di cucina (mediterranea, asiatica, americana, ecc.)
4. Tecniche di preparazione

Restituisci solo i numeri delle 3 ricette più diverse in formato JSON: {"selected_recipes": [X, Y, Z]}
"""

    user_prompt = f"""
Ho {len(recipes)} ricette che soddisfano già i requisiti nutrizionali e dietetici (target CHO: {preferences.target_cho}g
{"✓ Vegano " if preferences.vegan else ""}{"✓ Vegetariano " if preferences.vegetarian else ""}{"✓ Senza glutine " if preferences.gluten_free else ""}{"✓ Senza lattosio" if preferences.lactose_free else ""}).

Aiutami a selezionare le 3 ricette più diverse tra loro:

{all_recipes_text}

Restituisci SOLO i numeri delle 3 ricette più diverse in formato JSON.
"""

    # Chiama l'LLM
    model_name = "gpt-3.5-turbo"  # o il modello che preferisci
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(temperature=0, model_name=model_name,
                     openai_api_key=api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])

    # Esegui la chain
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({})

    # Estrai il JSON dalla risposta
    print(f"Risposta LLM per selezione ricette: {response}")
    json_match = re.search(r'\{.*\}', response, re.DOTALL)

    if json_match:
        try:
            result = json.loads(json_match.group())
            selected_indices = result.get("selected_recipes", [])

            # Verifica che gli indici siano validi (iniziano da 1 nel prompt)
            valid_indices = [
                i-1 for i in selected_indices if 1 <= i <= len(recipes)]

            if valid_indices:
                # Restituisci le ricette selezionate
                return [recipes[i] for i in valid_indices[:3]]

        except Exception as e:
            print(f"Errore nel parsing della risposta LLM: {e}")

    # Fallback: se l'LLM fallisce, usa l'approccio basato su regole
    print("Fallback alla selezione basata su regole")
    return select_diverse_recipes_rule_based(recipes)


def select_diverse_recipes_rule_based(recipes):
    """
    Selezione basata su regole come fallback se l'LLM fallisce.
    """
    # Ordina per CHO (più vicino al target)
    sorted_recipes = sorted(recipes, key=lambda r: abs(
        r.total_cho - preferences.target_cho))

    # Selezione basata su diversità di ingredienti principali
    selected_recipes = []
    recipe_types = set()
    main_ingredients = set()

    for recipe in sorted_recipes:
        # Estrai il tipo di ricetta dal nome
        recipe_type = recipe.name.split()[0].lower()

        # Trova l'ingrediente principale
        main_ingredient = ""
        if recipe.ingredients:
            main_ingredient = max(recipe.ingredients,
                                  key=lambda ing: ing.cho_contribution).name

        # Se questo tipo o ingrediente principale è già stato usato, salta
        if (recipe_type in recipe_types) or (main_ingredient in main_ingredients and main_ingredient):
            continue

        # Altrimenti, aggiungi questa ricetta
        recipe_types.add(recipe_type)
        if main_ingredient:
            main_ingredients.add(main_ingredient)
        selected_recipes.append(recipe)

        # Se abbiamo abbastanza ricette, termina
        if len(selected_recipes) >= 3:
            break

    # Se non abbiamo abbastanza ricette diverse, prendi le migliori
    if len(selected_recipes) < 3:
        for recipe in sorted_recipes:
            if recipe not in selected_recipes:
                selected_recipes.append(recipe)
                if len(selected_recipes) >= 3:
                    break

    return selected_recipes
