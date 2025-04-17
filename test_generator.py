# test_generator.py

# Imports
import pandas as pd
from agents.generator_agent import generate_recipes_agent
from model_schema import UserPreferences, GraphState, IngredientInfo
from utils import normalize_name  # Or wherever it's defined
import os

# Load ingredient data


def load_ingredient_data(csv_path="data/ingredients.csv"):
    # ... (Adapt your data loading logic here) ...
    try:
        # Adjust path if necessary, based on script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, csv_path)
        df = pd.read_csv(full_path, index_col="name",
                         dtype={'cho_per_100g': float})
        # Handle potential duplicate names (similar to app.py)
        if df.index.duplicated().any():
            print("Attenzione (loaders): Nome ingrediente duplicato. Verrà sovrascritto.")
            df = df[~df.index.duplicated(keep='last')]
        data = {name: IngredientInfo(name=name, **row.to_dict())
                for name, row in df.iterrows()}
        print(
            f"--- Caricamento Info Base completato ({len(data)} ingredienti) --- ")
        return data
    except Exception as e:
        print(f"Errore caricamento: {e}")
        return {}


# Create user preferences
preferences = UserPreferences(
    vegan=False, vegetarian=True, gluten_free=False, lactose_free=False,  target_cho=50)

# Load data and create initial state
ingredient_data = load_ingredient_data()
initial_state = GraphState(user_preferences=preferences, available_ingredients_data=ingredient_data,
                           generated_recipes=[], normalize_function=normalize_name, error_message=None)

# Call the agent
final_state = generate_recipes_agent(initial_state)

# Print results
if final_state.get("error_message"):
    print(f"Error: {final_state['error_message']}")
if final_state.get("generated_recipes"):
    for recipe in final_state["generated_recipes"]:
        print(f"Generated Recipe: {recipe.name}")
        # Print more recipe details if needed
        # print(recipe)
else:
    print("No recipes generated.")
