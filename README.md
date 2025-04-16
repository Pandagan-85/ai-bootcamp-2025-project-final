# Sistema Generatore di Ricette Personalizzate

## 📋 Panoramica

Questo progetto implementa un sistema avanzato di generazione di ricette personalizzate basato su LLM che crea ricette con un contenuto specifico di carboidrati (CHO) rispettando preferenze dietetiche come vegano, vegetariano, senza glutine e senza lattosio.

Il sistema utilizza un'architettura basata su agenti orchestrati da LangGraph per gestire il flusso completo: dalla generazione delle ricette alla loro verifica, fino alla formattazione dell'output finale per l'utente.

## 🎯 Obiettivo del progetto

Creare ricette bilanciate e personalizzate che soddisfino specifiche esigenze dietetiche e nutrizionali, con particolare attenzione al contenuto di carboidrati (CHO), permettendo di generare pasti adatti a persone con restrizioni alimentari o che necessitano di controllare l'apporto di carboidrati.

## 🏗️ Architettura del sistema

Il sistema è costruito con un'architettura modulare che utilizza il framework LangGraph per orchestrare il flusso di generazione delle ricette attraverso diversi agenti specializzati.

### Flusso di lavoro generale

```
Caricamento dati → Generazione ricette → Verifica ricette → Formattazione output
```

Il grafo di LangGraph implementa questo flusso con la possibilità di percorsi alternativi in caso di errori o altre condizioni specifiche.

## 📊 Modelli di dati

Il sistema utilizza diversi modelli di dati definiti in `model_schema.py`:

- `UserPreferences`: Le preferenze dell'utente (target CHO, vegano, vegetariano, senza glutine, senza lattosio)
- `IngredientInfo`: Informazioni nutrizionali e dietetiche di un ingrediente
- `Recipe`: Una ricetta con ingredienti e proprietà dietetiche
- `RecipeIngredient`: Un ingrediente nella ricetta con la sua quantità
- `CalculatedIngredient`: Un ingrediente con contributo nutrizionale calcolato
- `FinalRecipeOption`: Una ricetta completa con tutti i valori nutrizionali calcolati
- `GraphState`: Lo stato che viene passato tra i nodi del grafo LangGraph

## 🧠 Agenti e loro funzioni

### Generator Agent (`generator_agent.py`)

**Responsabilità**: Generare ricette da zero utilizzando LLM (GPT-3.5/4) che soddisfino i criteri nutrizionali e le preferenze dietetiche.

**Dettagli implementativi**:

- Utilizza un prompt ingegnerizzato per guidare l'LLM nella creazione di ricette con target CHO specifico
- Genera più ricette concorrenti utilizzando ThreadPoolExecutor per parallelizzare le richieste
- Implementa un meccanismo di retry in caso di problemi con l'LLM
- Filtra gli ingredienti disponibili in base alle preferenze dietetiche
- Verifica la validità delle ricette generate e calcola il contenuto nutrizionale

**Strategia di generazione**:

1. Crea un elenco di ingredienti validi basati sulle preferenze dietetiche
2. Invia il prompt all'LLM con tutte le informazioni pertinenti
3. Estrae e valida l'output JSON dall'LLM
4. Calcola i contributi CHO di ogni ingrediente
5. Verifica che la ricetta soddisfi i requisiti minimi

### Verifier Agent (`verifier_agent.py`)

**Responsabilità**: Verificare e ottimizzare le ricette generate per garantire che soddisfino i requisiti e bilanciarne la composizione.

**Funzioni chiave**:

- `validate_recipe_ingredients`: Verifica che tutti gli ingredienti esistano nel database
- `adjust_recipe_cho`: Modifica le quantità degli ingredienti per raggiungere il target CHO
- `balance_cho_distribution`: Migliora la distribuzione dei CHO tra gli ingredienti

**Verifiche eseguite**:

1. Validità degli ingredienti
2. Range di CHO entro la tolleranza
3. Compatibilità con le preferenze dietetiche
4. Bilanciamento della distribuzione dei CHO
5. Numero minimo di ingredienti per una ricetta ragionevole

L'agente effettua anche aggiustamenti proattivi:

- Bilancia ricette con ingredienti troppo dominanti
- Regola le quantità per avvicinarsi al target CHO
- Identifica e risolve potenziali problemi di distribuzione dei nutrienti

### Formatter Agent (`formatter_agent.py`)

**Responsabilità**: Formattare l'output finale in un formato leggibile per l'utente.

**Caratteristiche**:

- Genera output Markdown strutturato
- Presenta le ricette con dettagli nutrizionali completi
- Gestisce sia i casi di successo che di fallimento
- Evidenzia le caratteristiche dietetiche delle ricette
- Fornisce messaggi di errore informativi e suggerimenti quando necessario

## 🔄 Workflow di esecuzione

Il flusso di esecuzione è gestito da `workflow.py` che crea un grafo LangGraph con nodi e decisioni:

1. **generate_recipes**: Punto di ingresso che genera ricette da zero
2. **decide_after_generation**: Decisione su come procedere dopo la generazione
   - Se ci sono ricette generate → passa alla verifica
   - Se c'è un errore o nessuna ricetta → passa direttamente all'output
3. **verify_recipes**: Verifica e ottimizza le ricette generate
4. **format_output**: Formatta i risultati per la presentazione

## 🚀 Utilizzo

Il programma si esegue da linea di comando tramite `main.py` con parametri specifici:

```bash
python main.py 80 --vegan --gluten_free
```

Parametri:

- Il primo argomento è il target CHO in grammi (obbligatorio)
- Flag opzionali: `--vegan`, `--vegetarian`, `--gluten_free`, `--lactose_free`

## 📂 Struttura dei file

- `main.py`: Punto di ingresso dell'applicazione
- `model_schema.py`: Definizione dei modelli di dati
- `workflow.py`: Configurazione del grafo LangGraph
- `utils.py`: Funzioni di utilità comuni
- `loaders.py`: Caricamento dei dati degli ingredienti e delle ricette
- `agents/`: Directory contenente gli agenti specializzati
  - `generator_agent.py`: Generazione di ricette
  - `verifier_agent.py`: Verifica e ottimizzazione delle ricette
  - `formatter_agent.py`: Formattazione dell'output
  - `retriever_agent.py`: Recupero ricette (non utilizzato nel flusso principale)

## 🔍 Dettagli tecnici aggiuntivi

### Calcolo dei contributi nutrizionali

Il sistema calcola i contributi nutrizionali di ogni ingrediente in base alla quantità utilizzata:

```python
cho_contribution = (quantity_g * cho_per_100g) / 100.0
```

Questo viene applicato anche per calorie, proteine, grassi e fibre quando disponibili.

### Tolleranza e bilanciamento

- Tolleranza CHO: La deviazione accettabile dal target CHO (generalmente ±10g)
- Bilanciamento dei CHO: Nessun singolo ingrediente dovrebbe contribuire più del 90% del CHO totale
- Quantità minime: Il sistema garantisce che le modifiche mantengano quantità realistiche per ogni ingrediente

### Strategie di ottimizzazione delle ricette

1. **Aggiustamento proporzionale**: Scala tutti gli ingredienti ricchi di CHO
2. **Aggiustamento mirato**: Modifica solo l'ingrediente principale
3. **Bilanciamento**: Redistribuisce i CHO tra più ingredienti
4. **Sostituzione**: Aggiunge nuovi ingredienti se necessario

## 💡 Punti di forza del sistema

- **Altamente personalizzabile**: Target CHO e preferenze dietetiche configurabili
- **Robustezza**: Gestione degli errori a più livelli
- **Scalabilità**: Elaborazione parallela per la generazione di ricette
- **Ottimizzazione automatica**: Le ricette vengono automaticamente aggiustate per soddisfare i requisiti
- **Output completo**: Le ricette includono tutti i dettagli nutrizionali
- **Esperienza utente**: Output formattato in modo chiaro e leggibile

## 🔄 Flusso di elaborazione dettagliato

1. L'utente specifica target CHO e preferenze dietetiche
2. Il sistema carica il database degli ingredienti
3. Il Generator Agent crea ricette multiple in parallelo
4. Il Verifier Agent controlla e ottimizza le ricette
5. Il sistema seleziona le migliori ricette verificate
6. Il Formatter Agent crea un output leggibile
7. L'output viene presentato all'utente
