# model_schema.py

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, TypedDict

# --- Input Utente ---


class UserPreferences(BaseModel):
    target_cho: float = Field(description="Target CHO in grams")
    vegan: bool = Field(description="Vegan preference")
    vegetarian: bool = Field(description="Vegetarian preference")
    gluten_free: bool = Field(description="Gluten-Free preference")
    lactose_free: bool = Field(description="Lactose-Free preference")

# --- Strutture Dati ---


class IngredientInfo(BaseModel):
    name: str
    cho_per_100g: float
    is_vegan: bool
    is_vegetarian: bool
    is_gluten_free: bool
    is_lactose_free: bool


class RecipeIngredient(BaseModel):
    name: str
    quantity_g: float


class Recipe(BaseModel):
    name: str
    ingredients: List[RecipeIngredient]
    # Questi flag nel dataset sono utili per il pre-filtering
    is_vegan_recipe: bool
    is_vegetarian_recipe: bool
    is_gluten_free_recipe: bool
    is_lactose_free_recipe: bool
    # Il CHO totale può essere pre-calcolato o calcolato al volo
    initial_total_cho: Optional[float] = None


class CalculatedIngredient(BaseModel):
    name: str
    quantity_g: float
    cho_contribution: float  # CHO apportato da questo ingrediente specifico


class FinalRecipeOption(BaseModel):
    name: str
    ingredients: List[CalculatedIngredient]
    total_cho: float
    # Aggiungiamo i flag per chiarezza nell'output finale
    is_vegan: bool
    is_vegetarian: bool
    is_gluten_free: bool
    is_lactose_free: bool


# --- Stato del Grafo LangGraph ---
class GraphState(TypedDict):
    """
    Rappresenta lo stato del nostro grafo.
    Contiene tutti gli attributi che possono cambiare durante l'esecuzione.
    """
    user_preferences: UserPreferences
    # Dizionario per accesso rapido
    available_ingredients: Dict[str, IngredientInfo]
    initial_recipes: List[Recipe]  # Ricette recuperate inizialmente
    # Ricette modificate e analizzate
    processed_recipes: List[FinalRecipeOption]
    # Ricette finali dopo verifica
    final_verified_recipes: List[FinalRecipeOption]
    error_message: Optional[str]  # Per gestire eventuali errori nel flusso
