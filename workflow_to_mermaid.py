# workflow_to_mermaid.py
#
# Script per generare un diagramma Mermaid del workflow di generazione ricette
# Può essere eseguito separatamente o importato come modulo

import os
import sys
import inspect
from typing import Dict, Any, Tuple, List, Optional, Set
from workflow import create_workflow


def extract_graph_structure(app):
    """
    Estrae la struttura del grafo dall'applicazione LangGraph compilata.

    Args:
        app: L'applicazione LangGraph compilata

    Returns:
        Tuple contenente nodi, archi e condizioni
    """
    try:
        # Ottieni il grafo sottostante
        graph = app.get_graph()

        # Estrai i nodi
        nodes = list(graph._graph.nodes())

        # Estrai gli archi diretti e le loro condizioni
        edges = []
        edge_conditions = {}

        for source, targets in graph._graph.adj.items():
            for target, edge_data in targets.items():
                edges.append((source, target))
                if 'condition' in edge_data:
                    edge_conditions[(source, target)] = edge_data.get(
                        'condition')

        # Estrai i nodi condizionali
        conditional_nodes = set()
        for source, target in edges:
            if (source, target) in edge_conditions:
                conditional_nodes.add(source)

        return nodes, edges, edge_conditions, conditional_nodes

    except Exception as e:
        print(f"Errore nell'estrazione della struttura del grafo: {e}")
        # Fallback a struttura basata sul codice sorgente in workflow.py
        return extract_graph_structure_from_code()


def extract_graph_structure_from_code():
    """
    Fallback: estrae la struttura del grafo analizzando il codice sorgente di workflow.py.
    Usato solo se l'estrazione diretta fallisce.

    Returns:
        Tuple contenente nodi, archi e condizioni
    """
    # Struttura predefinita basata sulla lettura del workflow.py
    nodes = ["retrieve_recipes", "modify_recipes",
             "verify_recipes", "format_output", "END"]
    edges = [
        ("retrieve_recipes", "modify_recipes"),
        ("retrieve_recipes", "format_output"),
        ("modify_recipes", "verify_recipes"),
        ("modify_recipes", "format_output"),
        ("verify_recipes", "format_output"),
        ("format_output", "END")
    ]
    edge_conditions = {
        ("retrieve_recipes", "modify_recipes"): "decide_after_retrieval (ricette trovate)",
        ("retrieve_recipes", "format_output"): "decide_after_retrieval (errore/nessuna ricetta)",
        ("modify_recipes", "verify_recipes"): "decide_after_modification (ricette processate)",
        ("modify_recipes", "format_output"): "decide_after_modification (errore/nessuna ricetta)"
    }
    conditional_nodes = {"retrieve_recipes", "modify_recipes"}

    return nodes, edges, edge_conditions, conditional_nodes


def generate_mermaid_code(nodes, edges, edge_conditions, conditional_nodes):
    """
    Genera il codice Mermaid basato sulla struttura del grafo.

    Args:
        nodes: Lista di nomi dei nodi
        edges: Lista di tuple (source, target) rappresentanti gli archi
        edge_conditions: Dizionario di condizioni per gli archi
        conditional_nodes: Set di nodi con decisioni condizionali

    Returns:
        Stringa contenente il codice Mermaid
    """
    mermaid_code = ["flowchart TD"]

    # Aggiungi stili
    mermaid_code.append(
        "    classDef node fill:#f9f9f9,stroke:#333,stroke-width:1px")
    mermaid_code.append(
        "    classDef decision fill:#e1f5fe,stroke:#0288d1,stroke-width:1px")
    mermaid_code.append(
        "    classDef errorPath fill:#ffebee,stroke:#c62828,stroke-width:1px")
    mermaid_code.append(
        "    classDef successPath fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px")
    mermaid_code.append("")

    # Aggiungi nodo iniziale
    mermaid_code.append("    A[Inizio] --> " + nodes[0])

    # Gestisci i nodi condizionali
    node_ids = {}
    for node in nodes:
        if node == "END":
            node_ids[node] = "Z"
        elif node in conditional_nodes:
            # Per i nodi condizionali, crea un nodo e una decisione separata
            node_ids[node] = node[0].upper()
            decision_id = f"{node[0].upper()}D"
            mermaid_code.append(
                f"    {node_ids[node]}[{node}] --> {decision_id}{{{node}}}")
        else:
            node_ids[node] = node[0].upper()
            mermaid_code.append(f"    {node_ids[node]}[{node}]")

    # Aggiungi gli archi
    for source, target in edges:
        if source in conditional_nodes:
            # Per archi da nodi condizionali, usa il nodo decisione
            decision_id = f"{node_ids[source]}D"
            edge_label = ""

            if (source, target) in edge_conditions:
                condition = edge_conditions[(source, target)]
                if "errore" in condition.lower() or "nessuna" in condition.lower():
                    edge_label = "|Errore o nessuna ricetta|"
                    # Aggiungi classe error path al nodo target
                    if target != "END":
                        mermaid_code.append(
                            f"    {node_ids[target]}:::errorPath")
                else:
                    edge_label = "|Ricette trovate|"

            mermaid_code.append(
                f"    {decision_id} -->{edge_label} {node_ids[target]}")
        elif source != "END" and target != "END":
            mermaid_code.append(
                f"    {node_ids[source]} --> {node_ids[target]}")

    # Aggiungi node classi
    agent_nodes = [node for node in nodes if node !=
                   "END" and node not in conditional_nodes]
    if agent_nodes:
        mermaid_code.append("")
        mermaid_code.append("    subgraph Agenti")
        for node in agent_nodes:
            mermaid_code.append(f"        {node_ids[node]}:::node")
        mermaid_code.append("    end")

    # Aggiungi decision classi
    if conditional_nodes:
        mermaid_code.append("")
        mermaid_code.append("    subgraph Decisioni")
        for node in conditional_nodes:
            decision_id = f"{node_ids[node]}D"
            mermaid_code.append(f"        {decision_id}:::decision")
        mermaid_code.append("    end")

    return "\n".join(mermaid_code)


def generate_workflow_diagram(output_dir="."):
    """
    Genera un diagramma Mermaid del workflow e lo salva in formato MD e HTML.

    Args:
        output_dir: Directory dove salvare i file di output

    Returns:
        Tuple con i percorsi dei file generati (markdown, html)
    """
    print("\n--- Generazione Diagramma Workflow ---")

    # Crea il workflow
    print("Creazione workflow...")
    app = create_workflow()

    # Estrai struttura del grafo
    print("Estrazione struttura grafo...")
    nodes, edges, edge_conditions, conditional_nodes = extract_graph_structure(
        app)

    # Genera codice Mermaid
    print("Generazione codice Mermaid...")
    mermaid_code = generate_mermaid_code(
        nodes, edges, edge_conditions, conditional_nodes)

    # Stampa il codice generato
    print("\nCodice Mermaid generato:")
    print(mermaid_code)

    # Salva in file .md
    md_output_path = os.path.join(output_dir, "workflow_diagram.md")
    md_content = f"# Workflow LangGraph per Generazione Ricette\n\n```mermaid\n{mermaid_code}\n```"

    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nDiagramma Mermaid salvato in: {md_output_path}")

    # Salva anche come HTML per visualizzazione in browser
    html_output_path = os.path.join(output_dir, "workflow_diagram.html")
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Workflow Diagram - Generazione Ricette</title>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{ startOnLoad: true }});
        </script>
    </head>
    <body>
        <h1>Workflow Diagram - Generazione Ricette</h1>
        <div class="mermaid">
        {mermaid_code}
        </div>
    </body>
    </html>
    """

    with open(html_output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Diagramma HTML salvato in: {html_output_path}")

    return md_output_path, html_output_path


if __name__ == "__main__":
    # Se eseguito direttamente, genera il diagramma
    try:
        md_path, html_path = generate_workflow_diagram()
        print(f"\nDiagramma generato con successo!")
        print(f"- Markdown: {md_path}")
        print(f"- HTML: {html_path}")
        print("\nPuoi aprire il file HTML in un browser per visualizzare il diagramma interattivo.")
    except Exception as e:
        print(f"Errore durante la generazione del diagramma: {e}")
        sys.exit(1)
