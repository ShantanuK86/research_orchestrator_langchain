"""
Supervisor Agent — LangChain LCEL chain
prompt | ChatGoogleGenerativeAI | StrOutputParser
"""
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.core.models import ResearchState
from backend.core.config import GEMINI_MODEL

SYSTEM = """You are the Supervisor agent in a LangGraph multi-agent research pipeline.
Analyze the research topic and produce a structured plan.

Respond ONLY in this exact JSON (no markdown fences):
{{
  "topic": "cleaned, specific research topic",
  "plan": "2-3 sentence research strategy",
  "search_queries": ["angle 1", "angle 2", "angle 3"],
  "quality_criteria": ["criterion 1", "criterion 2", "criterion 3"],
  "max_iterations": 3
}}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Research topic: {query}"),
])

async def supervisor_node(state: ResearchState, api_key: str) -> dict:
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, google_api_key=api_key,
        temperature=0.3, max_output_tokens=600,
    )
    chain = prompt | llm | StrOutputParser()
    raw = await chain.ainvoke({"query": state.query})
    raw = raw.replace("```json", "").replace("```", "").strip()

    events = [
        {"agent": "supervisor", "type": "thought", "message": f'Query received: "{state.query}"'},
        {"agent": "supervisor", "type": "log", "message": "LCEL chain: PromptTemplate | Gemini | StrOutputParser"},
    ]

    try:
        data = json.loads(raw)
        events.append({"agent": "supervisor", "type": "log",
            "message": f"Plan ready — {len(data['search_queries'])} angles · criteria: {', '.join(data['quality_criteria'])}"})
        return {
            "topic": data.get("topic", state.query),
            "plan": data.get("plan", ""),
            "search_queries": data.get("search_queries", [state.query]),
            "quality_criteria": data.get("quality_criteria", ["accuracy", "depth", "relevance"]),
            "max_iterations": data.get("max_iterations", 3),
            "events": events,
        }
    except Exception:
        events.append({"agent": "supervisor", "type": "log", "message": "Fallback plan activated", "thinking": True})
        return {
            "topic": state.query, "plan": "Direct research with iterative refinement.",
            "search_queries": [state.query, f"{state.query} analysis", f"{state.query} trends"],
            "quality_criteria": ["accuracy", "comprehensiveness", "recency"],
            "max_iterations": 3, "events": events,
        }
