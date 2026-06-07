from typing import TypedDict, Annotated
import operator


class ResearchState(TypedDict):
    """
    Shared state that flows through every node in the LangGraph.
    Each agent reads from and writes to this dict.
    """
    # Input
    query: str
    api_key: str
    max_iterations: int

    # Supervisor output
    topic: str
    plan: str
    search_queries: list[str]
    quality_criteria: list[str]

    # Research loop
    research_content: str
    iteration: int
    approved: bool
    quality_score: int
    gaps: list[str]
    critique: str

    # Final output
    final_report: str
    total_tokens: int

    # SSE event queue — agents append events here for streaming
    events: Annotated[list[dict], operator.add]
