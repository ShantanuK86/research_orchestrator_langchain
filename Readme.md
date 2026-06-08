# 🔬 Research Orchestrator v2

> A proper multi-agent research system built on **LangGraph StateGraph** + **LangChain LCEL** — not just prompt chaining, actual agent orchestration with conditional routing, iterative self-critique, and live SSE streaming.

![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-0.4-purple?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-0.3-orange?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-gray?style=flat-square)

---

## why this exists

Most "AI research tools" are just one big mega-prompt. They hallucinate, miss contradictions, and confidently produce absolute garbage.

I wanted to build something better. This system uses a real **LangGraph StateGraph** where *five* specialized agents relentlessly check each other's work in a loop until the output is bulletproof:

```text
START
  │
  ▼
Supervisor ──────────────────────────────────────────────────┐
  │  builds plan, sets search angles                         │
  ▼                                                          │
Search Agent ◄───────────────────────────────────────────────┤
  │  synthesizes knowledge via LangChain messages            │
  ▼                                                          │
Fact-Checker ────────────────────────────────────────────────┤
  │  ruthlessly hunts down hallucinations & flags bad claims │
  ▼                                                          │
Critic Agent ──── score < 7 ──► back to Search (with gaps)   │
  │                                                          │
  │  score ≥ 7                                               │
  ▼                                                          │
Writer Agent                                                 │
  │  LCEL chain: prompt | LLM | StrOutputParser              │
  ▼                                                          │
END ◄─────────────────────────────────────────────────────────┘
```

The Critic loop runs up to 3 times. Every agent decision streams live to the UI via SSE. The graph topology is defined once in `research_graph.py` — LangGraph handles state merging, node routing, and conditional edges.

---

## what's actually different from v1

The [previous version](https://github.com/shantanuk86/research-orchestrator) used a raw `httpx` client and a manual `while` loop. This one uses the real frameworks:

| Thing              | v1 (old)                        | v2 (this)                              |
|--------------------|----------------------------------|----------------------------------------|
| Orchestration      | `while not approved:` loop       | `LangGraph StateGraph`                 |
| LLM calls          | Raw `httpx.AsyncClient.post()`   | `ChatGoogleGenerativeAI` via LangChain |
| Prompt building    | f-strings                        | `ChatPromptTemplate` + LCEL chains     |
| Message format     | Dict payloads                    | `HumanMessage` / `SystemMessage`       |
| Agent routing      | Manual if/else                   | `add_conditional_edges`                |
| State management   | Local variables                  | `ResearchState` Pydantic model         |
| Output parsing     | `json.loads(raw)`                | `StrOutputParser` in LCEL chain        |

---

## project structure

```
research_orchestrator_langchain/
│
├── backend/
│   │
│   ├── graph/
│   │   └── research_graph.py       # LangGraph StateGraph definition
│   │                                 # add_node, add_edge, add_conditional_edges
│   │
│   ├── agents/
│   │   ├── supervisor.py           # ChatPromptTemplate | Gemini | StrOutputParser
│   │   ├── search_agent.py         # [SystemMessage, HumanMessage] → llm.ainvoke()
│   │   ├── fact_checker.py         # The hallucination hunter!
│   │   ├── critic.py               # LCEL chain → JSON quality score
│   │   └── writer.py               # LCEL chain → final Markdown report
│   │
│   ├── api/
│   │   └── routes.py               # FastAPI SSE endpoint, graph.astream()
│   │
│   └── core/
│       ├── config.py               # env vars
│       └── models.py               # ResearchState (LangGraph state), API schemas
│
├── frontend/
│   ├── index.html                  # LangGraph visualizer + agent log
│   ├── css/style.css
│   └── js/
│       ├── api.js                  # SSE stream reader
│       ├── ui.js                   # DOM: graph nodes, score rings, loop indicator
│       └── main.js                 # wires SSE events → UI state
│
├── main.py                         # FastAPI app + static file serving
├── requirements.txt                # langchain, langgraph, langchain-google-genai
├── Procfile                        # Railway deployment
├── runtime.txt                     # python-3.12.13
└── .env.example
```

---

## running it

**1. clone**
```bash
git clone https://github.com/shantanuk86/research_orchestrator_langchain.git
cd research_orchestrator_langchain
```

**2. install — use uv, it's faster**
```bash
brew install uv                          
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

**3. configure**
```bash
cp .env.example .env
# set GEMINI_API_KEY=AIza...
```

Free key at [aistudio.google.com](https://aistudio.google.com) — no credit card.

**4. run**
```bash
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000`.

---

## how the LangGraph pieces fit together

**State** — `ResearchState` in `core/models.py` is a Pydantic model shared across all nodes. LangGraph merges node output dicts back into state automatically. The `events` field uses `Annotated[list[dict], operator.add]` so each node can append logs without overwriting.

**Graph** — defined in `backend/graph/research_graph.py`:
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

# The key bit — conditional routing after critic
graph.add_conditional_edges(
    "critic",
    route_after_critic,          # returns "writer" or "search"
    {"writer": "writer", "search": "search"},
)
```

**LCEL chains** — each agent builds a chain like:
```python
chain = ChatPromptTemplate.from_messages([...]) | ChatGoogleGenerativeAI(...) | StrOutputParser()
result = await chain.ainvoke({"query": state.query})
```

**Streaming** — `routes.py` uses `graph.astream(init_state, stream_mode="updates")` which yields node outputs as they complete. Each node's `events` list is drained and forwarded to the frontend as SSE.

---

## API

### `POST /api/v1/research/stream`

Streams SSE events. Each line is `data: <json>\n\n`.

**Request:**
```json
{
  "query": "Impact of LangGraph on production AI systems",
  "api_key": "AIza...",
  "max_iterations": 3
}
```

**Event format:**
```json
{
  "agent":   "supervisor | search | critic | writer | system",
  "type":    "log | status | metric | score | result | thought",
  "message": "what's happening",
  "thinking": false,
  "data":    {}
}
```

Special event types:
- `score` — from critic node, includes `{score: int, approve: bool, iteration: int}`
- `result` — final report, includes full markdown and run metadata
- `thought` — agent internal reasoning, shown as thought bubble in UI

Stream ends with `data: [DONE]`.

### `GET /api/v1/health`
```json
{ "status": "ok", "service": "research-orchestrator", "framework": "LangGraph 0.4" }
```

---

## deploying to Railway

```bash
# push to GitHub first
git init && git add . && git commit -m "feat: langgraph research orchestrator"
gh repo create research_orchestrator_langchain --public --source=. --push

# then:
# 1. railway.app → New Project → deploy from GitHub
# 2. Variables tab → add GEMINI_API_KEY
# 3. deploy — public URL in ~2 min
```

Railway reads `Procfile` automatically:
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## what I want to add

- [ ] Tavily/Serper web search tool for the Search Agent (real-time data)
- [ ] LangSmith tracing — one env var: `LANGCHAIN_TRACING_V2=true`
- [ ] Token streaming from Writer via `llm.astream()`
- [ ] `MemorySaver` checkpointing for persistent run history
- [x] Fifth node: Fact-Checker between Search and Critic (Shipped! 🚀)
- [ ] Export report as PDF

---

[![GitHub](https://img.shields.io/badge/GitHub-shantanukumar-black?style=flat-square&logo=github)](https://github.com/shantanuk86)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-shantanuk86-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/shantanuk86)

---

MIT license.