"""
Definizione del flusso di lavoro per la generazione di ricette.

Questo modulo definisce il grafo di esecuzione (workflow) utilizzando LangGraph
per orchestrare gli agenti coinvolti nella generazione e verifica di ricette personalizzate.
Implementa l'approccio "Generate then Fix" con un generatore semplificato e un verificatore potenziato.
"""
from typing import Literal

from langgraph.graph import StateGraph, END

# Importa lo schema dello stato e le funzioni dei nodi
from model_schema import GraphState

from agents.generator_agent import generate_recipes_agent
from agents.verifier_agent import verifier_agent
from agents.formatter_agent import format_output_agent


# --- Funzioni Decisionali per il Routing Condizionale ---

def decide_after_generation(state: GraphState) -> Literal["verify_recipes", "format_output"]:
    """
    Funzione decisionale che determina il percorso da seguire dopo la generazione delle ricette.

    Questa funzione prende decisioni in base allo stato attuale del grafo:
    - Se ci sono errori critici, va direttamente all'output
    - Se non sono state generate ricette, va direttamente all'output
    - Altrimenti, procede con la verifica delle ricette

    Args:
        state: Lo stato attuale del grafo (GraphState)

    Returns:
        Una stringa che indica il nodo successivo da eseguire:
        - "verify_recipes": per procedere con la verifica delle ricette
        - "format_output": per saltare la verifica e passare direttamente all'output
    """
    print("--- DECISIONE: Dopo Generazione Ricette ---")

    # Controlla errori specifici della fase di generazione o se la lista è vuota
    if state.get("error_message") and ("LLM" in state["error_message"] or "API Key" in state["error_message"]):
        print(
            f"Percorso: Errore Generazione ({state.get('error_message')}) -> Format Output")
        return "format_output"

    if not state.get("generated_recipes"):
        print("Percorso: Nessuna ricetta generata -> Format Output")
        # Assicurati che ci sia un messaggio di errore appropriato se non già presente
        if not state.get("error_message"):
            state['error_message'] = "Nessuna ricetta è stata generata con successo."
        return "format_output"
    else:
        print("Percorso: Ricette generate trovate -> Verifica Ricette")
        return "verify_recipes"


# --- Creazione del Grafo ---

def create_workflow() -> StateGraph:
    """
    Crea e configura il grafo LangGraph per il processo di generazione ricette.

    Implementa il flusso "Generate then Fix" con:
    1. Generator Agent: Generazione creativa con vincoli minimi
    2. Verifier Agent: Matching, validazione e ottimizzazione
    3. Formatter Agent: Presentazione dei risultati
    """
    print("--- Creazione Grafo Workflow ---")
    # Inizializza il grafo con lo stato definito
    workflow = StateGraph(GraphState)

    # 1. Aggiungi i Nodi al grafo
    print("Aggiunta nodo: generate_recipes")
    workflow.add_node("generate_recipes", generate_recipes_agent)

    print("Aggiunta nodo: verify_recipes")
    workflow.add_node("verify_recipes", verifier_agent)

    print("Aggiunta nodo: format_output")
    workflow.add_node("format_output", format_output_agent)

    # 2. Definisci il Punto di Ingresso
    print("Impostazione punto di ingresso: generate_recipes")
    workflow.set_entry_point("generate_recipes")

    # 3. Aggiungi le Connessioni Condizionali
    print("Aggiunta edge condizionale: generate_recipes -> (decide_after_generation)")
    workflow.add_conditional_edges(
        "generate_recipes",           # Nodo di partenza
        decide_after_generation,      # Funzione che decide il prossimo passo
        {                             # Mapping: output della funzione -> nome del nodo target
            "verify_recipes": "verify_recipes",
            "format_output": "format_output"
        }
    )

    # 4. Aggiungi le Connessioni Dirette
    # Dopo la verifica, andiamo sempre alla formattazione (che gestirà successo/fallimento)
    print("Aggiunta edge: verify_recipes -> format_output")
    workflow.add_edge("verify_recipes", "format_output")

    # Il nodo finale porta automaticamente a END
    print("Aggiunta edge: format_output -> END")
    workflow.add_edge("format_output", END)

    # 5. Compila il grafo in un'applicazione eseguibile
    print("--- Compilazione Grafo ---")
    app = workflow.compile()
    print("--- Grafo Compilato con Successo ---")
    return app
