# model_schema.py
from typing import List, Dict, Optional, TypedDict, Any, Callable
# Aggiungi ConfigDict se usi Pydantic V2
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
import faiss  # Importa faiss per type hint (opzionale)
from sentence_transformers import SentenceTransformer

# --- Input Utente ---


class UserPreferences(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)
    target_cho: float = Field(description="Target CHO in grams")
    vegan: bool = Field(description="Vegan preference")
    vegetarian: bool = Field(description="Vegetarian preference")
    gluten_free: bool = Field(description="Gluten-Free preference")
    lactose_free: bool = Field(description="Lactose-Free preference")

# --- Strutture Dati ---


class IngredientInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    # Rendi opzionale? O gestisci default in caricamento
    cho_per_100g: Optional[float] = None
    calories_per_100g: Optional[float] = None
    protein_g_per_100g: Optional[float] = None
    fat_g_per_100g: Optional[float] = None
    fiber_g_per_100g: Optional[float] = None
    is_vegan: bool = False
    is_vegetarian: bool = False
    is_gluten_free: bool = False
    is_lactose_free: bool = False


class RecipeIngredient(BaseModel):
    """Ingrediente in una ricetta con quantità."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(
        description="Nome ingrediente (come trovato nel DB dopo matching)")
    quantity_g: float = Field(description="Quantità in grammi")


class CalculatedIngredient(RecipeIngredient):
    """Ingrediente con contributi nutrizionali calcolati."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    cho_per_100g: Optional[float] = None
    cho_contribution: Optional[float] = None
    # Aggiungi altri campi opzionali se li calcoli in utils
    calories_per_100g: Optional[float] = None
    calories_contribution: Optional[float] = None
    protein_g_per_100g: Optional[float] = None
    protein_contribution_g: Optional[float] = None
    fat_g_per_100g: Optional[float] = None
    fat_contribution_g: Optional[float] = None
    fiber_g_per_100g: Optional[float] = None
    fiber_contribution_g: Optional[float] = None
    # Aggiungi flag dietetici per riferimento
    is_vegan: bool = False
    is_vegetarian: bool = False
    is_gluten_free: bool = False
    is_lactose_free: bool = False
    # Opzionale: tieni traccia del nome LLM originale
    original_llm_name: Optional[str] = None


class FinalRecipeOption(BaseModel):
    """Rappresenta una ricetta finale validata e pronta per l'utente."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    description: Optional[str] = None
    ingredients: List[CalculatedIngredient]  # Ora usa CalculatedIngredient
    # Rendi opzionale se non sempre calcolato?
    total_cho: Optional[float] = None
    total_calories: Optional[float] = None
    total_protein_g: Optional[float] = None
    total_fat_g: Optional[float] = None
    total_fiber_g: Optional[float] = None
    is_vegan: bool
    is_vegetarian: bool
    is_gluten_free: bool
    is_lactose_free: bool
    instructions: Optional[List[str]] = None
    # Opzionale: aggiungi score o deviazione per ranking
    cho_deviation_percent: Optional[float] = None

# --- Stato del Grafo LangGraph ---


class GraphState(TypedDict, total=False):  # Usa total=False per opzionalità
    """
    Rappresenta lo stato del grafo LangGraph con FAISS.
    """
    # Input e dati caricati
    user_preferences: UserPreferences
    available_ingredients_data: Dict[str,
                                     IngredientInfo]  # Dizionario nome->info
    embedding_model: SentenceTransformer                 # Modello SBERT per query
    # Funzione normalizzazione
    normalize_function: Callable[[str], str]
    faiss_index: faiss.Index                             # Indice FAISS caricato
    # Mapping indice -> nome
    index_to_name_mapping: List[str]

    # Risultati intermedi e finali
    # Ricette da LLM (post-parsing/validazione iniziale)
    generated_recipes: List[FinalRecipeOption]
    # Ricette dopo verifica/aggiustamento/ranking
    final_verified_recipes: List[FinalRecipeOption]

    # Gestione errori e output
    error_message: Optional[str]
    final_output: Optional[str]

    # Rimuovi campi relativi a embedding NumPy
    # ingredient_names_list: List[str]      <- Rimosso
    # ingredient_embeddings: np.ndarray     <- Rimosso
