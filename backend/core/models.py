from pydantic import BaseModel, Field
from typing import Optional, Annotated
import operator


# ── API schemas ───────────────────────────────────────────
class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=500)
    api_key: Optional[str] = None
    max_iterations: Optional[int] = Field(None, ge=1, le=5)


class AgentEvent(BaseModel):
    agent: str
    type: str
    message: str
    thinking: bool = False
    data: Optional[dict] = None


# ── LangGraph state ───────────────────────────────────────
class ResearchState(BaseModel):
    query: str                      = ""
    topic: str                      = ""
    plan: str                       = ""
    search_queries: list[str]       = []
    quality_criteria: list[str]     = []
    max_iterations: int             = 3
    research_content: str           = ""
    quality_score: int              = 0
    approved: bool                  = False
    gaps: list[str]                 = []
    improvements: list[str]         = []
    critique_text: str              = ""
    fact_check_report: str          = ""
    final_report: str               = ""
    iteration: int                  = 0
    total_tokens: int               = 0
    events: Annotated[list[dict], operator.add] = []
