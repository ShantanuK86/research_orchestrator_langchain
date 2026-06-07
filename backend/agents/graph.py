"""
LangGraph StateGraph — the actual orchestration engine.
Defines nodes, edges, and conditional routing between agents.
"""
from langgraph.graph import StateGraph, END
from backend.agents.supervisor import supervisor_node
from backend.agents.search_agent import search_node
from backend.agents.critic import critic_node
from backend.agents.writer import writer_node
from backend.core.config import MAX_ITERATIONS


def increment_iteration(state: dict) -> dict:
    """Increment the iteration counter between search and critic."""
    return {**state, "iteration": state.get("iteration", 0) + 1}


def should_continue(state: dict) -> str:
    """
    Conditional edge: after Critic, decide whether to loop back to Search
    or proceed to Writer.
    - Loop if: not approved AND iteration < max_iterations
    - Write if: approved OR iteration >= max_iterations
    """
    approved = state.get("approved", False)
    iteration = state.get("iteration", 1)
    max_iter = state.get("plan", {}).get("max_iterations", MAX_ITERATIONS)

    if approved or iteration >= max_iter:
        return "write"
    return "search"


def build_graph() -> StateGraph:
    """
    Graph topology:
    supervisor → search → critic ─(loop)─► search
                                └─(done)─► writer → END
    """
    g = StateGraph(dict)

    g.add_node("supervisor", supervisor_node)
    g.add_node("increment",  increment_iteration)
    g.add_node("search",     search_node)
    g.add_node("critic",     critic_node)
    g.add_node("writer",     writer_node)

    g.set_entry_point("supervisor")
    g.add_edge("supervisor", "increment")
    g.add_edge("increment",  "search")
    g.add_edge("search",     "critic")

    g.add_conditional_edges(
        "critic",
        should_continue,
        {"search": "increment", "write": "writer"},
    )

    g.add_edge("writer", END)
    return g.compile()


# Singleton compiled graph
research_graph = build_graph()
