"""
Definizione del flusso di lavoro per la generazione di ricette.

Questo modulo definisce il grafo di esecuzione (workflow) utilizzando LangGraph
per orchestrare i vari agenti coinvolti nella generazione di ricette personalizzate.
Il workflow è strutturato come un grafo diretto con nodi (agenti) e archi (transizioni)
che definiscono il flusso di esecuzione.
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
    - Se ci sono errori critici (relativi all'LLM o all'API Key), va direttamente all'output
    - Se non sono state generate ricette, va direttamente all'output
    - Altrimenti, procede con la verifica delle ricette

    Args:
        state: Lo stato attuale del grafo (GraphState)

    Returns:
        Una stringa che indica il nodo successivo da eseguire:
        - "verify_recipes": per procedere con la verifica delle ricette
        - "format_output": per saltare la verifica e passare direttamente all'output

    Note:
        - Assicura che ci sia un messaggio di errore appropriato se non sono state generate ricette
        - Il percorso scelto viene registrato nel log per il debug
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

    Questo metodo definisce:
    1. I nodi del grafo (agenti che elaborano lo stato)
    2. Il punto di ingresso (da dove inizia l'esecuzione)
    3. Le connessioni condizionali (decisioni basate sullo stato)
    4. Le connessioni dirette (transizioni semplici)
    """
    print("--- Creazione Grafo Workflow ---")
    # Inizializza il grafo con lo stato definito
    workflow = StateGraph(GraphState)

    # 1. Aggiungi i Nodi al grafo
    # Ogni nodo è una funzione che prende lo stato e restituisce lo stato modificato
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


# Esempio di come usare la funzione (verrà chiamato da main.py)
if __name__ == '__main__':
    compiled_app = create_workflow()
    # Stampa informazioni sul grafo compilato (opzionale)
    # print(compiled_app.get_graph().print_ascii()) # Stampa ASCII del grafo
