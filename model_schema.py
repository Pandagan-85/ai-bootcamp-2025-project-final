"""
Modello dei dati per il sistema di generazione ricette.

Questo modulo definisce tutte le classi Pydantic utilizzate per rappresentare i dati nel sistema di generazione di ricette, dalla definizione delle preferenze utente alla rappresentazione degli ingredienti, delle ricette e del loro contenuto nutrizionale.
L'intero flusso di lavoro utilizza queste strutture per garantire la coerenza dei dati.
"""

from typing import List, Dict, Optional, TypedDict
from pydantic import BaseModel, Field


# --- Input Utente ---


class UserPreferences(BaseModel):
    """
    Rappresenta le preferenze dell'utente per il sistema di generazione ricette.

    Queste preferenze includono il target di carboidrati desiderato in grammi e i flag per le varie restrizioni dietetiche (vegano, vegetariano, senza glutine, senza lattosio).
    """
    target_cho: float = Field(description="Target CHO in grams")
    vegan: bool = Field(description="Vegan preference")
    vegetarian: bool = Field(description="Vegetarian preference")
    gluten_free: bool = Field(description="Gluten-Free preference")
    lactose_free: bool = Field(description="Lactose-Free preference")

# --- Strutture Dati ---


class IngredientInfo(BaseModel):
    """
    Contiene le informazioni nutrizionali e le proprietà dietetiche di un ingrediente.

    Ogni ingrediente ha un nome, un valore di carboidrati per 100g, e vari flag che indicano le sue proprietà (vegano, vegetariano, senza glutine, senza lattosio).
    Opzionalmente può includere informazioni su calorie, proteine, grassi e fibre. (forse per le proteine sfrutteremo il dato per compensare il loro valore cho 0 e inserirli in ricette a basso contenuto di cho in maniera equilibrata)
    """
    name: str
    cho_per_100g: float
    calories_per_100g: Optional[float] = None
    protein_per_100g: Optional[float] = None
    fat_per_100g: Optional[float] = None
    fiber_per_100g: Optional[float] = None
    # Gruppo alimentare (es. cereali, verdure, ecc.)
    food_group: Optional[str] = None
    is_vegan: bool
    is_vegetarian: bool
    is_gluten_free: bool
    is_lactose_free: bool


class RecipeIngredient(BaseModel):
    """
    Rappresenta un ingrediente all'interno di una ricetta con la sua quantità.

    Questa classe è una versione semplificata di IngredientInfo che contiene solo
    il nome dell'ingrediente e la quantità in grammi utilizzata nella ricetta.
    """
    name: str  # Nome dell'ingrediente (deve corrispondere a un nome in IngredientInfo)
    quantity_g: float


class Recipe(BaseModel):
    """
    Rappresenta una ricetta nel suo formato base (prima del calcolo nutrizionale).

    Contiene il nome della ricetta, la lista degli ingredienti con le loro quantità, e i flag dietetici. Il valore CHO totale può essere precalcolato o calcolato dinamicamente in base agli ingredienti.
    """
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
    """
    Estende RecipeIngredient con i dati nutrizionali calcolati per l'ingrediente.

    Oltre al nome e alla quantità, include il contributo effettivo in termini di carboidrati, calorie, proteine, grassi e fibre che l'ingrediente apporta alla ricetta in base alla sua quantità. (in questo modo possiamo anche riportarlo nell'output finale)
    """
    name: str
    quantity_g: float
    cho_contribution: float  # CHO apportato da questo ingrediente specifico
    calories_contribution: Optional[float] = None
    protein_contribution_g: Optional[float] = None
    fat_contribution_g: Optional[float] = None
    fiber_contribution_g: Optional[float] = None


class FinalRecipeOption(BaseModel):
    """
    Rappresenta una ricetta completa con tutti i dettagli nutrizionali calcolati.

    Questa è la struttura della ricetta finale dopo la verifica e il calcolo
    nutrizionale. Include i totali nutrizionali, gli ingredienti con i loro contributi, i flag dietetici e opzionalmente le istruzioni e la descrizione.
    """
    name: str
    ingredients: List[CalculatedIngredient]
    total_cho: float
    total_calories: Optional[float] = None
    total_protein_g: Optional[float] = None
    total_fat_g: Optional[float] = None
    total_fiber_g: Optional[float] = None
    # Aggiungiamo i flag per chiarezza nell'output finale
    is_vegan: bool
    is_vegetarian: bool
    is_gluten_free: bool
    is_lactose_free: bool
    instructions: Optional[List[str]] = None
    description: Optional[str] = None


# --- Stato del Grafo LangGraph ---
class GraphState(TypedDict):
    """
     Rappresenta lo stato del grafo LangGraph durante l'esecuzione del workflow.

     Questa classe è un dizionario tipizzato che contiene tutti gli attributi
     che possono cambiare durante l'esecuzione del workflow. Viene passato da
     un nodo all'altro del grafo e contiene sia i dati di input che i risultati intermedi e finali.
     """
    user_preferences: UserPreferences
    # Dizionario per accesso rapido
    available_ingredients: Dict[str, IngredientInfo]
    # Ricette recuperate inizialmente (opzionale nel nuovo flusso)
    initial_recipes: List[Recipe]
    # Ricette generate (nuovo nel flusso aggiornato)
    generated_recipes: List[FinalRecipeOption]
    # Ricette finali dopo verifica
    final_verified_recipes: List[FinalRecipeOption]
    error_message: Optional[str]  # Per gestire eventuali errori nel flusso
    final_output: Optional[str]   # Output formattato finale
    # Dizionario con HTML per le icone (opzionale)
    img_dict: Optional[Dict] = None
