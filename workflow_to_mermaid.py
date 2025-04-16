# workflow_to_mermaid.py
#
# Script per generare un diagramma Mermaid del workflow di generazione ricette
# Utilizza le funzionalità native di LangGraph per generare un diagramma aggiornato

import os
import sys
import io
from typing import Dict, Any, Tuple, List, Optional
from workflow import create_workflow


def generate_workflow_diagram(output_dir="."):
    """
    Genera un diagramma Mermaid del workflow e lo salva in formato MD e HTML.
    Utilizza le funzionalità native di LangGraph.

    Args:
        output_dir: Directory dove salvare i file di output

    Returns:
        Tuple con i percorsi dei file generati (markdown, html)
    """
    print("\n--- Generazione Diagramma Workflow ---")

    # Crea il workflow
    print("Creazione workflow...")
    app = create_workflow()

    # Utilizza il metodo nativo di LangGraph per ottenere la rappresentazione Mermaid
    print("Generazione codice Mermaid tramite LangGraph...")

    # Cattura l'output del print_ascii() che normalmente va a stdout
    buffer = io.StringIO()

    # Prova prima con il grafo direttamente, poi con altri metodi se necessario
    try:
        # Metodo 1: Prova a ottenere direttamente la rappresentazione Mermaid
        graph = app.get_graph()

        # In LangGraph recenti, c'è un metodo get_mermaid()
        if hasattr(graph, 'get_mermaid'):
            mermaid_code = graph.get_mermaid()
            print("Utilizzato metodo get_mermaid()")

        # Altrimenti proviamo con to_mermaid()
        elif hasattr(graph, 'to_mermaid'):
            mermaid_code = graph.to_mermaid()
            print("Utilizzato metodo to_mermaid()")

        # Se nessuno dei due è disponibile, usiamo print_ascii() che in alcune versioni
        # permette di specificare un formato ('mermaid')
        else:
            # Verifica se print_ascii accetta il parametro 'format'
            import inspect
            sig = inspect.signature(graph.print_ascii)
            if 'format' in sig.parameters:
                # Cattura l'output di print_ascii
                old_stdout = sys.stdout
                sys.stdout = buffer
                graph.print_ascii(format='mermaid')
                sys.stdout = old_stdout
                mermaid_code = buffer.getvalue().strip()
                print("Utilizzato print_ascii(format='mermaid')")
            else:
                # Usa la visualizzazione ASCII come fallback e convertiamo in Mermaid
                old_stdout = sys.stdout
                sys.stdout = buffer
                graph.print_ascii()
                sys.stdout = old_stdout
                ascii_graph = buffer.getvalue().strip()

                # Converti da ASCII a Mermaid (versione semplificata)
                mermaid_code = convert_ascii_to_mermaid(ascii_graph)
                print("Utilizzato print_ascii() con conversione manuale")

    except Exception as e:
        print(f"Errore nell'ottenere la rappresentazione Mermaid: {e}")
        # Fallback al vecchio metodo
        print("Fallback alla generazione manuale del diagramma...")
        mermaid_code = generate_manual_mermaid(app)

    # Se ancora non abbiamo un codice Mermaid, usiamo la versione manuale
    if not mermaid_code:
        print("Generazione manuale del diagramma come fallback...")
        mermaid_code = generate_manual_mermaid(app)

    # Migliora lo stile del codice Mermaid
    mermaid_code = enhance_mermaid_style(mermaid_code)

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
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
            }}
            h1 {{
                color: #333;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <h1>Workflow Diagram - Generazione Ricette</h1>
        <p>Diagramma generato automaticamente dal workflow LangGraph</p>
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


def enhance_mermaid_style(mermaid_code):
    """
    Migliora lo stile del codice Mermaid generato automaticamente.

    Args:
        mermaid_code: Codice Mermaid originale

    Returns:
        Codice Mermaid migliorato con stili e formattazione
    """
    # Se il codice non inizia con "flowchart" o "graph", aggiungi l'intestazione
    if not mermaid_code.strip().startswith(("flowchart", "graph")):
        mermaid_code = "flowchart TD\n" + mermaid_code

    # Dividi il codice in righe
    lines = mermaid_code.strip().split("\n")
    enhanced_lines = []

    # Aggiungi la prima riga (intestazione del diagramma)
    enhanced_lines.append(lines[0])

    # Aggiungi stili
    enhanced_lines.append(
        "    classDef node fill:#f9f9f9,stroke:#333,stroke-width:1px")
    enhanced_lines.append(
        "    classDef decision fill:#e1f5fe,stroke:#0288d1,stroke-width:1px")
    enhanced_lines.append(
        "    classDef errorPath fill:#ffebee,stroke:#c62828,stroke-width:1px")
    enhanced_lines.append(
        "    classDef successPath fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px")
    enhanced_lines.append(
        "    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px")

    # Aggiungi le altre righe
    for line in lines[1:]:
        # Migliora la formattazione aggiungendo spazi per indentazione
        if not line.startswith("    "):
            line = "    " + line
        enhanced_lines.append(line)

    # Se ci sono nodi condizionali, aggiungi classi per la decisione
    # In questo caso dovremmo analizzare il grafo per trovarli, ma è più complesso
    # Useremo un semplice euristica basata sui nomi
    decision_nodes = []
    for line in lines:
        if "decide_after_" in line:
            parts = line.split("-->")
            if len(parts) > 0:
                node_id = parts[0].strip()
                if node_id and node_id not in decision_nodes:
                    decision_nodes.append(node_id)

    # Aggiungi classi per i nodi trovati
    if decision_nodes:
        enhanced_lines.append("")
        for node in decision_nodes:
            enhanced_lines.append(f"    {node}:::decision")

    return "\n".join(enhanced_lines)


def convert_ascii_to_mermaid(ascii_graph):
    """
    Converte una rappresentazione ASCII del grafo in codice Mermaid.
    Questa è una versione semplificata e potrebbe richiedere miglioramenti.

    Args:
        ascii_graph: Rappresentazione ASCII del grafo

    Returns:
        Codice Mermaid equivalente
    """
    # Inizializza l'output Mermaid
    mermaid_lines = ["flowchart TD"]

    # Analizza il grafo ASCII
    lines = ascii_graph.strip().split('\n')

    # Estrai nodi e archi
    nodes = set()
    edges = []

    for line in lines:
        if "->" in line:
            parts = line.split("->")
            if len(parts) >= 2:
                source = parts[0].strip()
                target = parts[1].strip()
                nodes.add(source)
                nodes.add(target)
                edges.append((source, target))

    # Genera il codice Mermaid
    for node in nodes:
        mermaid_lines.append(f"    {node}[{node}]")

    for source, target in edges:
        mermaid_lines.append(f"    {source} --> {target}")

    return "\n".join(mermaid_lines)


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
    nodes = ["generate_recipes", "verify_recipes", "format_output", "END"]
    edges = [
        ("generate_recipes", "verify_recipes"),
        ("generate_recipes", "format_output"),
        ("verify_recipes", "format_output"),
        ("format_output", "END")
    ]
    edge_conditions = {
        ("generate_recipes", "verify_recipes"): "decide_after_generation (ricette generate)",
        ("generate_recipes", "format_output"): "decide_after_generation (errore/nessuna ricetta)"
    }
    conditional_nodes = {"generate_recipes"}

    return nodes, edges, edge_conditions, conditional_nodes


def generate_manual_mermaid(app):
    """
    Genera manualmente il codice Mermaid basato sulla struttura del grafo.
    Usato come fallback se i metodi nativi di LangGraph non funzionano.

    Args:
        app: L'applicazione LangGraph compilata

    Returns:
        Stringa contenente il codice Mermaid
    """
    nodes, edges, edge_conditions, conditional_nodes = extract_graph_structure(
        app)

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
