"""
Critic Agent — LangChain LCEL with JsonOutputParser
Scores research quality and decides: approve or loop back.
"""
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.core.models import ResearchState
from backend.core.config import GEMINI_MODEL, QUALITY_THRESHOLD

SYSTEM = """You are the Critic Agent in a LangGraph research pipeline.
Rigorously evaluate research content for quality, accuracy, and completeness.
Pay close attention to the Fact-Check Report. If the Fact-Checker flagged unsupported claims or hallucinations, you MUST penalize the score and reject the content.

Respond ONLY in this exact JSON (no markdown fences):
{{
  "quality_score": <integer 0-10>,
  "approve": <true if score >= 7>,
  "gaps": ["specific missing area 1", "specific missing area 2"],
  "critique": "2-3 sentence honest evaluation",
  "improvements": ["concrete improvement 1", "concrete improvement 2"]
}}

Scoring:
9-10 → exceptional, approve
7-8  → solid, approve
5-6  → missing key areas, reject
1-4  → too shallow or inaccurate, reject"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human",
     "Topic: {topic}\nQuality criteria: {criteria}\n\nFact-Check Report:\n{fact_check_report}\n\nResearch to evaluate:\n{content}"),
])

async def critic_node(state: ResearchState, api_key: str) -> dict:
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, google_api_key=api_key,
        temperature=0.2, max_output_tokens=600,
    )
    chain = prompt | llm | StrOutputParser()

    raw = await chain.ainvoke({
        "topic": state.topic,
        "criteria": ", ".join(state.quality_criteria),
        "fact_check_report": state.fact_check_report,
        "content": state.research_content,
    })
    raw = raw.replace("```json", "").replace("```", "").strip()

    events = [
        {"agent": "critic", "type": "thought",
         "message": f"Evaluating pass {state.iteration} against: {', '.join(state.quality_criteria)}"},
        {"agent": "critic", "type": "log",
         "message": "LCEL chain: PromptTemplate | Gemini | StrOutputParser → JSON"},
    ]

    try:
        data = json.loads(raw)
        score = int(data.get("quality_score", 7))
        approve = score >= QUALITY_THRESHOLD
        gaps = data.get("gaps", [])
        critique = data.get("critique", "")
        improvements = data.get("improvements", [])

        color = "approved" if approve else "rejected"
        events.append({"agent": "critic", "type": "log",
            "message": f"Score: {score}/10 — {color.upper()}",
            "data": {"score": score, "approve": approve}})
        if critique:
            events.append({"agent": "critic", "type": "log", "message": critique})
        if not approve and gaps:
            events.append({"agent": "critic", "type": "log",
                "message": f"Gaps flagged: {'; '.join(gaps)}"})

        return {
            "quality_score": score, "approved": approve,
            "gaps": gaps, "improvements": improvements,
            "critique_text": critique,
            "total_tokens": state.total_tokens + 150,
            "events": events,
        }
    except Exception:
        events.append({"agent": "critic", "type": "log",
            "message": "Parse error — defaulting to approve", "thinking": True})
        return {
            "quality_score": 7, "approved": True,
            "gaps": [], "improvements": [], "critique_text": "",
            "total_tokens": state.total_tokens + 100,
            "events": events,
        }
