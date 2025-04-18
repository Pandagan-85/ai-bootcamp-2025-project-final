#!/usr/bin/env python3
# simple_workflow_diagram.py - Genera solo il diagramma hardcoded del workflow

import os


def generate_workflow_diagram(output_dir="."):
    """
    Genera un diagramma Mermaid dettagliato del workflow "Generate then Fix"
    e lo salva in formato MD e HTML.

    Args:
        output_dir: Directory dove salvare i file di output

    Returns:
        Tuple con i percorsi dei file generati (markdown, html)
    """
    print("\n--- Generazione Diagramma Workflow ---")

    # Codice Mermaid hardcoded
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

    # Salva in file .md
    md_output_path = os.path.join(output_dir, "workflow_diagram.md")
    md_content = f"# Workflow LangGraph per Generazione Ricette\n\n```mermaid\n{mermaid_code}\n```"

    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Diagramma Markdown salvato in: {md_output_path}")

    # Salva in file .html
    html_output_path = os.path.join(output_dir, "workflow_diagram.html")
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Workflow Diagram - Generazione Ricette</title>
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
        <h1>Workflow Diagram - Generazione Ricette</h1>
        <p>Diagramma del sistema di generazione ricette "Generate then Fix" con architettura LangGraph</p>
        <div class="mermaid">
        {mermaid_code}
        </div>
        <div class="footer">
            <p>Sistema Generatore di Ricette Personalizzate - Approccio "Generate then Fix"</p>
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
        print("\nDiagramma generato con successo!")
        print(f"- Markdown: {md_path}")
        print(f"- HTML: {html_path}")
        print("\nPuoi aprire il file HTML in un browser per visualizzare il diagramma interattivo.")
    except Exception as e:
        print(f"Errore durante la generazione del diagramma: {e}")
        import sys
        sys.exit(1)
