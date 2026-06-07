"""
research_graph.py
─────────────────
The LangGraph StateGraph that orchestrates all agents.

Graph topology:
  START → supervisor → search → critic → [writer | search (loop)]
                                  ↑______________|  (conditional edge)

Conditional routing:
  After critic node:
    - approved OR iteration >= max  → "writer"
    - not approved AND can retry    → "search"  (loop back with gap context)
"""
from langgraph.graph import StateGraph, START, END
from backend.core.models import ResearchState
from backend.agents.supervisor import supervisor_node
from backend.agents.search_agent import search_node
from backend.agents.critic import critic_node
from backend.agents.writer import writer_node
from backend.core.config import MAX_ITERATIONS


def build_graph(api_key: str):
    """
    Build and compile the LangGraph StateGraph.
    Each node is a partial function with api_key injected.
    """

    # ── Node wrappers (inject api_key via closure) ────────
    async def _supervisor(state: ResearchState) -> dict:
        return await supervisor_node(state, api_key)

    async def _search(state: ResearchState) -> dict:
        # Increment iteration counter on each search pass
        new_iter = state.iteration + 1
        result = await search_node(
            ResearchState(**{**state.model_dump(), "iteration": new_iter}),
            api_key
        )
        result["iteration"] = new_iter
        return result

    async def _critic(state: ResearchState) -> dict:
        return await critic_node(state, api_key)

    async def _writer(state: ResearchState) -> dict:
        return await writer_node(state, api_key)

    # ── Conditional edge: should we loop or finish? ───────
    def route_after_critic(state: ResearchState) -> str:
        max_iter = min(state.max_iterations, MAX_ITERATIONS)
        if state.approved or state.iteration >= max_iter:
            return "writer"
        return "search"

    # ── Build graph ───────────────────────────────────────
    graph = StateGraph(ResearchState)

    graph.add_node("supervisor", _supervisor)
    graph.add_node("search",     _search)
    graph.add_node("critic",     _critic)
    graph.add_node("writer",     _writer)

    # Edges
    graph.add_edge(START,        "supervisor")
    graph.add_edge("supervisor", "search")
    graph.add_edge("search",     "critic")
    graph.add_edge("writer",     END)

    # Conditional edge — the core LangGraph routing logic
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"writer": "writer", "search": "search"},
    )

    return graph.compile()
