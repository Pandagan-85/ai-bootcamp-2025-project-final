import pandas as pd
import numpy as np
import json

# Carica il dataset degli ingredienti generato precedentemente
ingredients_df = pd.read_csv('data/ingredients.csv')

# Funzione per calcolare il CHO di una ricetta


def calculate_recipe_cho(ingredients, ingredients_df):
    total_cho = 0
    ingredients_with_cho = []
    for ing in ingredients:
        ingredient_row = ingredients_df[ingredients_df['name']
                                        == ing['name']].iloc[0]
        cho_per_100g = ingredient_row['cho_per_100g']
        quantity_g = ing['quantity_g']
        cho_contribution = (cho_per_100g * quantity_g) / 100
        total_cho += cho_contribution
        ingredients_with_cho.append({
            "name": ing['name'],
            "quantity_g": quantity_g,
            "cho_contribution": cho_contribution
        })
    return round(total_cho, 2), ingredients_with_cho

# Funzione per determinare i flag dietetici di una ricetta


def determine_dietary_flags(ingredients, ingredients_df):
    is_vegan_recipe = True
    is_vegetarian_recipe = True
    is_gluten_free_recipe = True
    is_lactose_free_recipe = True
    for ing in ingredients:
        ingredient_row = ingredients_df[ingredients_df['name']
                                        == ing['name']].iloc[0]
        if not ingredient_row['is_vegan']:
            is_vegan_recipe = False
        if not ingredient_row['is_vegetarian']:
            is_vegetarian_recipe = False
        if not ingredient_row['is_gluten_free']:
            is_gluten_free_recipe = False
        if not ingredient_row['is_lactose_free']:
            is_lactose_free_recipe = False
    return is_vegan_recipe, is_vegetarian_recipe, is_gluten_free_recipe, is_lactose_free_recipe


# Genera le ricette
recipes = []
for i in range(200):
    num_ingredients = np.random.randint(2, 6)  # Ricette con 2-5 ingredienti
    selected_ingredients = ingredients_df.sample(num_ingredients)
    recipe_ingredients = []
    for _, row in selected_ingredients.iterrows():
        # Quantità casuali tra 50 e 200g
        quantity_g = round(np.random.uniform(50, 200), 2)
        recipe_ingredients.append({
            "name": row['name'],
            "quantity_g": quantity_g
        })
    total_cho, ingredients_with_cho = calculate_recipe_cho(
        recipe_ingredients, ingredients_df)
    is_vegan_recipe, is_vegetarian_recipe, is_gluten_free_recipe, is_lactose_free_recipe = determine_dietary_flags(
        recipe_ingredients, ingredients_df)
    recipes.append({
        "name": f"Recipe {i+1}",
        "ingredients_json": json.dumps(recipe_ingredients),
        "is_vegan_recipe": is_vegan_recipe,
        "is_vegetarian_recipe": is_vegetarian_recipe,
        "is_gluten_free_recipe": is_gluten_free_recipe,
        "is_lactose_free_recipe": is_lactose_free_recipe,
        "total_cho": total_cho,
        "ingredients_with_cho_json": json.dumps(ingredients_with_cho)
    })

# Crea il DataFrame
recipes_df = pd.DataFrame(recipes)

# Salva in CSV
recipes_df.to_csv('data/recipes.csv', index=False)

print("File CSV 'data/recipes.csv' generato con successo.")
