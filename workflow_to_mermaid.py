#!/usr/bin/env python3
# workflow_diagrams.py - Genera diagrammi del workflow sia hardcoded che dinamici

import os
import sys
import io
import traceback


def generate_detailed_hardcoded_mermaid():
    """
    Genera un diagramma Mermaid dettagliato del workflow "Generate then Fix".
    Versione hardcoded che riproduce il layout e il livello di dettaglio desiderato.

    Returns:
        Stringa contenente il codice Mermaid dettagliato
    """
    mermaid_code = """flowchart TD
    %% Inizializzazione
    A[Caricamento Risorse] --> B[FAISS Index]
    A --> C[SentenceTransformer]
    A --> D[Ingredient DB]
    A --> E[User Preferences]
    B & C & D & E --> F[GraphState Initialization]
    
    %% Generator Agent
    F --> G[Generator Agent]
    G --> H[Generazione Ricette Creative]
    H --> I[Conversione a FinalRecipeOption]
    I --> K{decide_after_generation}
    
    %% Routing Decision
    K -->|Ricette Valide| L[Verifier Agent]
    K -->|Errore/No Ricette| X[Formatter Agent]
    
    %% Verifier Agent
    L --> M[Fase 1: Matching & Validazione]
    M --> O[Fase 2: Ottimizzazione CHO]
    O --> T[Fase 3: Verifica Finale & Selezione]
    T --> X
    
    %% Formatter Agent
    X --> Y[Formattazione HTML]
    Y --> Z[Output Finale]
    
    classDef blue fill:#ddf,stroke:#33f,stroke-width:2px
    classDef green fill:#dfd,stroke:#3a3,stroke-width:2px
    classDef orange fill:#fed,stroke:#f93,stroke-width:2px
    classDef yellow fill:#ffd,stroke:#ff3,stroke-width:2px
    classDef purple fill:#edf,stroke:#93f,stroke-width:2px
    
    class A,B,C,D,E,F blue
    class G,H,I green
    class K yellow
    class L,M,O,T orange
    class X,Y purple
    class Z gray"""

    return mermaid_code


def try_generate_dynamic_mermaid():
    """
    Tenta di generare un diagramma Mermaid estraendo la struttura direttamente da LangGraph.

    Returns:
        Tupla (mermaid_code, success_flag, error_message)
    """
    try:
        # Importa il workflow
        from workflow import create_workflow

        # Crea l'applicazione LangGraph
        print("Creazione workflow LangGraph...")
        app = create_workflow()

        # Tenta di ottenere il grafo
        graph = app.get_graph()

        # Opzioni per ottenere il codice Mermaid - in ordine di preferenza
        mermaid_code = None
        error_message = None

        # Opzione 1: get_mermaid()
        if hasattr(graph, 'get_mermaid'):
            try:
                mermaid_code = graph.get_mermaid()
                print("Utilizzato metodo get_mermaid()")
            except Exception as e:
                error_message = f"Errore con get_mermaid(): {e}"
                print(error_message)

        # Opzione 2: to_mermaid()
        if mermaid_code is None and hasattr(graph, 'to_mermaid'):
            try:
                mermaid_code = graph.to_mermaid()
                print("Utilizzato metodo to_mermaid()")
            except Exception as e:
                error_message = f"Errore con to_mermaid(): {e}"
                print(error_message)

        # Opzione 3: print_ascii(format='mermaid')
        if mermaid_code is None and hasattr(graph, 'print_ascii'):
            try:
                import inspect
                sig = inspect.signature(graph.print_ascii)
                if 'format' in sig.parameters:
                    # Cattura l'output di print_ascii
                    buffer = io.StringIO()
                    old_stdout = sys.stdout
                    sys.stdout = buffer
                    graph.print_ascii(format='mermaid')
                    sys.stdout = old_stdout
                    mermaid_code = buffer.getvalue().strip()
                    print("Utilizzato print_ascii(format='mermaid')")
            except Exception as e:
                error_message = f"Errore con print_ascii(format='mermaid'): {e}"
                print(error_message)

        # Verifica se abbiamo ottenuto un codice Mermaid
        if mermaid_code:
            # Migliora la formattazione del codice
            mermaid_code = improve_mermaid_formatting(mermaid_code)
            return mermaid_code, True, None
        else:
            return None, False, error_message or "Nessun metodo disponibile per ottenere il codice Mermaid"

    except Exception as e:
        error_message = f"Errore durante la generazione dinamica: {e}"
        print(error_message)
        traceback.print_exc()
        return None, False, error_message


def improve_mermaid_formatting(mermaid_code):
    """
    Migliora la formattazione del codice Mermaid generato automaticamente.

    Args:
        mermaid_code: Codice Mermaid originale

    Returns:
        Codice Mermaid con formattazione migliorata
    """
    # Assicura che inizi con flowchart TD
    if not mermaid_code.strip().startswith(("flowchart", "graph")):
        mermaid_code = "flowchart TD\n" + mermaid_code

    # Aggiungi stili
    style_lines = """
    classDef blue fill:#ddf,stroke:#33f,stroke-width:2px
    classDef green fill:#dfd,stroke:#3a3,stroke-width:2px
    classDef orange fill:#fed,stroke:#f93,stroke-width:2px
    classDef yellow fill:#ffd,stroke:#ff3,stroke-width:2px
    classDef purple fill:#edf,stroke:#93f,stroke-width:2px
    classDef decision fill:#ffebee,stroke:#c62828,stroke-width:1px
    """

    # Aggiungi gli stili se non sono già presenti
    if "classDef" not in mermaid_code:
        mermaid_code += style_lines

    return mermaid_code


def generate_html_content(mermaid_code, title, subtitle):
    """
    Genera il contenuto HTML per visualizzare un diagramma Mermaid.

    Args:
        mermaid_code: Codice Mermaid da visualizzare
        title: Titolo principale della pagina
        subtitle: Sottotitolo descrittivo

    Returns:
        Stringa contenente il codice HTML completo
    """
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{ 
                startOnLoad: true,
                theme: 'default',
                flowchart: {{
                    useMaxWidth: false,
                    curve: 'basis'
                }}
            }});
        </script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1 {{
                color: #333;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }}
            .mermaid {{
                text-align: center;
                margin: 30px 0;
            }}
            .footer {{
                margin-top: 40px;
                font-size: 0.9em;
                color: #666;
                text-align: center;
                border-top: 1px solid #eee;
                padding-top: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p>{subtitle}</p>
        <div class="mermaid">
        {mermaid_code}
        </div>
        <div class="footer">
            <p>Sistema Generatore di Ricette Personalizzate - Approccio "Generate then Fix"</p>
        </div>
    </body>
    </html>
    """


def generate_workflow_diagrams(output_dir="."):
    """
    Genera sia il diagramma hardcoded che quello dinamico del workflow.

    Args:
        output_dir: Directory dove salvare i file di output

    Returns:
        Dizionario con i percorsi dei file generati e info sul successo
    """
    print("\n--- Generazione Diagrammi Workflow ---")
    results = {"hardcoded": {}, "dynamic": {}}

    # 1. Genera il diagramma hardcoded (sempre funzionante)
    print("\n=== Generazione Diagramma Hardcoded ===")
    hardcoded_mermaid = generate_detailed_hardcoded_mermaid()

    # Salva in file .md
    hardcoded_md_path = os.path.join(
        output_dir, "workflow_diagram_hardcoded.md")
    md_content = f"# Workflow LangGraph per Generazione Ricette (Hardcoded)\n\n```mermaid\n{hardcoded_mermaid}\n```"

    with open(hardcoded_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Diagramma Markdown (hardcoded) salvato in: {hardcoded_md_path}")

    # Salva in file .html
    hardcoded_html_path = os.path.join(
        output_dir, "workflow_diagram_hardcoded.html")
    html_content = generate_html_content(
        hardcoded_mermaid,
        "Workflow Diagram - Generazione Ricette (Hardcoded)",
        "Diagramma predefinito del sistema di generazione ricette 'Generate then Fix'"
    )

    with open(hardcoded_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Diagramma HTML (hardcoded) salvato in: {hardcoded_html_path}")

    results["hardcoded"] = {
        "md_path": hardcoded_md_path,
        "html_path": hardcoded_html_path,
        "success": True
    }

    # 2. Tenta di generare il diagramma dinamico (da LangGraph)
    print("\n=== Generazione Diagramma Dinamico ===")
    dynamic_mermaid, success, error_message = try_generate_dynamic_mermaid()

    if success:
        # Salva in file .md
        dynamic_md_path = os.path.join(
            output_dir, "workflow_diagram_dynamic.md")
        md_content = f"# Workflow LangGraph per Generazione Ricette (Dinamico)\n\n```mermaid\n{dynamic_mermaid}\n```"

        with open(dynamic_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Diagramma Markdown (dinamico) salvato in: {dynamic_md_path}")

        # Salva in file .html
        dynamic_html_path = os.path.join(
            output_dir, "workflow_diagram_dynamic.html")
        html_content = generate_html_content(
            dynamic_mermaid,
            "Workflow Diagram - Generazione Ricette (Dinamico)",
            "Diagramma estratto dinamicamente dal workflow LangGraph"
        )

        with open(dynamic_html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Diagramma HTML (dinamico) salvato in: {dynamic_html_path}")

        results["dynamic"] = {
            "md_path": dynamic_md_path,
            "html_path": dynamic_html_path,
            "success": True
        }
    else:
        print(f"Generazione dinamica fallita: {error_message}")
        error_md_path = os.path.join(
            output_dir, "workflow_diagram_dynamic_error.md")

        with open(error_md_path, "w", encoding="utf-8") as f:
            f.write(
                f"# Errore nella Generazione Dinamica del Diagramma\n\n{error_message}")

        results["dynamic"] = {
            "md_path": error_md_path,
            "success": False,
            "error": error_message
        }

    return results


if __name__ == "__main__":
    # Se eseguito direttamente, genera entrambi i diagrammi
    try:
        results = generate_workflow_diagrams()

        print("\n=== Riepilogo Generazione ===")
        print("Diagramma Hardcoded:")
        print(f"- Markdown: {results['hardcoded']['md_path']}")
        print(f"- HTML: {results['hardcoded']['html_path']}")

        print("\nDiagramma Dinamico:")
        if results['dynamic']['success']:
            print(f"- Markdown: {results['dynamic']['md_path']}")
            print(f"- HTML: {results['dynamic']['html_path']}")
        else:
            print(f"- Generazione fallita: {results['dynamic']['error']}")
            print(f"- Log errore: {results['dynamic']['md_path']}")

        print("\nPuoi aprire i file HTML in un browser per visualizzare i diagrammi interattivi.")

    except Exception as e:
        print(f"Errore durante la generazione dei diagrammi: {e}")
        traceback.print_exc()
        sys.exit(1)
