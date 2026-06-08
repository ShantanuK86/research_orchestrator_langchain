# Architecture

## Overview

This system implements a **supervisor multi-agent pattern** using LangGraph's `StateGraph`. A single shared state object flows through an elite five-node squad, featuring a ruthless Fact-Checker and a conditional edge that creates an iterative critic loop before the final writer node runs.

```
┌──────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                           │
│                                                                  │
│  POST /api/v1/research/stream                                    │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              LangGraph StateGraph                        │   │
│  │                                                          │   │
│  │  ResearchState (Pydantic) ──────────────────────────┐   │   │
│  │       shared across all nodes via state merging     │   │   │
│  │                                                     │   │   │
│  │  START                                              │   │   │
│  │    │                                                │   │   │
│  │    ▼                                                │   │   │
│  │  [supervisor]  ChatPromptTemplate | LLM | Parser    │   │   │
│  │    │           → topic, plan, search_queries        │   │   │
│  │    ▼                                                │   │   │
│  │  [search]  ◄────────────────────────────────────┐  │   │   │
│  │    │       SystemMessage + HumanMessage → LLM   │  │   │   │
│  │    │       → research_content                   │  │   │   │
│  │    ▼                                            │  │   │   │
│  │  [fact_check]  ChatPromptTemplate | LLM | Parser│  │   │   │
│  │    │           → fact_check_report              │  │   │   │
│  │    ▼                                            │  │   │   │
│  │  [critic]      ChatPromptTemplate | LLM | Parser│  │   │   │
│  │    │           → quality_score, approved, gaps  │  │   │   │
│  │    │                                            │  │   │   │
│  │    ├── score < 7  ──  conditional_edge ─────────┘  │   │   │
│  │    │                  route_after_critic()          │   │   │
│  │    │                  returns "search"              │   │   │
│  │    │                                                │   │   │
│  │    └── score ≥ 7                                    │   │   │
│  │         │                                           │   │   │
│  │         ▼                                           │   │   │
│  │  [writer]      ChatPromptTemplate | LLM | Parser    │   │   │
│  │    │           → final_report (Markdown)            │   │   │
│  │    ▼                                                │   │   │
│  │  END                                                │   │   │
│  └──────────────────────────────────────────────────────┘   │   │
│                                                              │   │
│  graph.astream() → yields node updates → SSE events ────────┘   │
└──────────────────────────────────────────────────────────────────┘
         │
         │  text/event-stream  (one JSON object per line)
         ▼
┌─────────────────────────────────┐
│         Browser Frontend        │
│                                 │
│  api.js     SSE reader          │
│  ui.js      DOM updates         │
│  main.js    event router        │
│                                 │
│  Graph visualizer               │
│  Agent log                      │
│  Score rings (per iteration)    │
│  Loop-back indicator            │
└─────────────────────────────────┘
```

---

## LangGraph State

`ResearchState` in `backend/core/models.py` — a Pydantic `BaseModel` that LangGraph uses as the shared state object.

```python
class ResearchState(BaseModel):
    query: str                      = ""   # original user input
    topic: str                      = ""   # cleaned by supervisor
    plan: str                       = ""   # research strategy
    search_queries: list[str]       = []   # angles to cover
    quality_criteria: list[str]     = []   # what critic evaluates against
    max_iterations: int             = 3
    research_content: str           = ""   # current search output
    quality_score: int              = 0    # last critic score
    approved: bool                  = False
    gaps: list[str]                 = []   # critic-flagged gaps
    improvements: list[str]         = []
    critique_text: str              = ""
    fact_check_report: str          = ""   # hallucination flags
    final_report: str               = ""
    iteration: int                  = 0
    total_tokens: int               = 0
    events: Annotated[list[dict], operator.add] = []
```

The `events` field uses `operator.add` as the reducer — LangGraph appends node outputs rather than replacing them, so each node can push logs without overwriting other nodes' events.

---

## Graph Definition

`backend/graph/research_graph.py`

```python
graph = StateGraph(ResearchState)

graph.add_node("supervisor", _supervisor)
graph.add_node("search",     _search)
graph.add_node("fact_checker", _fact_checker)
graph.add_node("critic",     _critic)
graph.add_node("writer",     _writer)

graph.add_edge(START,        "supervisor")
graph.add_edge("supervisor", "search")
graph.add_edge("search",     "fact_checker")
graph.add_edge("fact_checker", "critic")
graph.add_edge("writer",     END)

graph.add_conditional_edges(
    "critic",
    route_after_critic,
    {"writer": "writer", "search": "search"},
)

return graph.compile()
```

`route_after_critic` is the decision function:
```python
def route_after_critic(state: ResearchState) -> str:
    if state.approved or state.iteration >= MAX_ITERATIONS:
        return "writer"
    return "search"
```

---

## Agent Details

### Supervisor
- **LangChain pattern:** `ChatPromptTemplate.from_messages | ChatGoogleGenerativeAI | StrOutputParser`
- **Input:** raw user query
- **Output state delta:** `topic`, `plan`, `search_queries`, `quality_criteria`, `max_iterations`
- **Temperature:** 0.3 (deterministic planning)

### Search Agent
- **LangChain pattern:** `[SystemMessage, HumanMessage]` → `llm.ainvoke(messages)`
- **Input:** `state.topic`, `state.search_queries`, `state.gaps` (on retry iterations)
- **Output state delta:** `research_content`, `total_tokens`
- **Temperature:** 0.7 (diverse synthesis)
- **Context-aware:** on iteration > 1, focuses prompt on critic-flagged gaps

### Fact-Checker Agent
- **LangChain pattern:** `ChatPromptTemplate | ChatGoogleGenerativeAI | StrOutputParser`
- **Input:** `state.topic`, `state.research_content`
- **Output state delta:** `fact_check_report`, `total_tokens`
- **Temperature:** 0.1 (icy-cold logic to hunt down hallucinations)
- **Role:** Ruthlessly dissects the Search Agent's output for unsupported claims before the Critic even sees it.

### Critic Agent
- **LangChain pattern:** `ChatPromptTemplate | ChatGoogleGenerativeAI | StrOutputParser` → `json.loads()`
- **Input:** `state.research_content`, `state.quality_criteria`, `state.fact_check_report`
- **Output state delta:** `quality_score`, `approved`, `gaps`, `improvements`, `critique_text`
- **Temperature:** 0.2 (consistent scoring)
- **Threshold:** configurable via `QUALITY_THRESHOLD` env var (default 7/10)

### Writer Agent
- **LangChain pattern:** `ChatPromptTemplate | ChatGoogleGenerativeAI | StrOutputParser`
- **Input:** `state.research_content`, `state.topic`, `state.iteration`, `state.quality_score`
- **Output state delta:** `final_report`, `total_tokens`
- **Temperature:** 0.6

---

## SSE Event Schema

All frontend updates arrive as SSE lines: `data: <json>\n\n`

```typescript
interface AgentEvent {
  agent:    "supervisor" | "search" | "fact_checker" | "critic" | "writer" | "system";
  type:     "log" | "status" | "metric" | "score" | "result" | "thought";
  message:  string;
  thinking: boolean;
  data?:    Record<string, any>;
}
```

Special `data` payloads by type:

| type     | data fields                                              |
|----------|----------------------------------------------------------|
| `status` | `{ label?: string }`                                     |
| `metric` | `{ iteration?: number, tokens?: number, elapsed?: number }` |
| `score`  | `{ score: number, approve: boolean, iteration: number }` |
| `result` | `{ topic, report, iterations, total_tokens, elapsed_seconds, quality_score }` |

Stream terminates with `data: [DONE]`.

---

## Environment Variables

| Variable             | Default              | Description                              |
|----------------------|----------------------|------------------------------------------|
| `GEMINI_API_KEY`     | —                    | Required. Get at aistudio.google.com     |
| `GEMINI_MODEL`       | `gemini-2.0-flash`   | Any Gemini model string                  |
| `MAX_ITERATIONS`     | `3`                  | Critic loop ceiling                      |
| `QUALITY_THRESHOLD`  | `7`                  | Min score to approve (0–10)              |

---

## Tech Stack

| Layer       | Technology                          | Why                                         |
|-------------|--------------------------------------|---------------------------------------------|
| Orchestration | LangGraph 0.4 StateGraph           | Proper graph topology, conditional edges    |
| LLM calls   | LangChain 0.3 + langchain-google-genai | LCEL chains, message types, output parsers |
| Backend     | FastAPI + Uvicorn                    | Async, SSE support, fast                    |
| Streaming   | Server-Sent Events                   | Simpler than WebSockets for unidirectional  |
| Frontend    | Vanilla JS ES Modules                | No bundler, zero deps, fast to load         |
| Validation  | Pydantic v2                          | LangGraph uses it natively                  |

---

## Differences from v1

v1 used a raw `httpx` client and manual `while` loop. This is not that.

| Concern            | v1                          | v2 (this)                           |
|--------------------|-----------------------------|--------------------------------------|
| Orchestration      | `while not approved:`       | `StateGraph` + `add_conditional_edges` |
| LLM interface      | Raw HTTP POST to Gemini API | `ChatGoogleGenerativeAI`             |
| Prompt building    | f-strings                   | `ChatPromptTemplate`                 |
| Message format     | Raw JSON dicts              | `HumanMessage`, `SystemMessage`      |
| Output parsing     | `json.loads(raw_text)`      | `StrOutputParser` in LCEL chain      |
| State passing      | Function arguments          | `ResearchState` Pydantic model       |
| Streaming          | Manual generator            | `graph.astream(stream_mode="updates")` |