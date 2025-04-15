# Workflow LangGraph per Generazione Ricette

```mermaid
flowchart TD
    classDef node fill:#f9f9f9,stroke:#333,stroke-width:1px
    classDef decision fill:#e1f5fe,stroke:#0288d1,stroke-width:1px
    classDef errorPath fill:#ffebee,stroke:#c62828,stroke-width:1px
    classDef successPath fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px

    A[Inizio] --> retrieve_recipes
    R[retrieve_recipes] --> RD{retrieve_recipes}
    M[modify_recipes] --> MD{modify_recipes}
    V[verify_recipes]
    F[format_output]
    RD -->|Ricette trovate| M
    F:::errorPath
    RD -->|Errore o nessuna ricetta| F
    MD -->|Ricette trovate| V
    F:::errorPath
    MD -->|Errore o nessuna ricetta| F
    V --> F

    subgraph Agenti
        V:::node
        F:::node
    end

    subgraph Decisioni
        RD:::decision
        MD:::decision
    end
```