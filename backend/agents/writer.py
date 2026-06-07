"""
Writer Agent — LangChain LCEL chain
Synthesizes the final Markdown report from approved research.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.core.models import ResearchState
from backend.core.config import GEMINI_MODEL

SYSTEM = """You are the Writer Agent in a LangGraph research pipeline.
Synthesize research findings into a polished, authoritative report.

Format in clean Markdown:
## [Compelling Title]
### Executive Summary
### Key Findings
### Deep Analysis
### Conclusion & Implications

Rules:
- **bold** for key terms and stats
- > blockquotes for critical insights
- `code` for technical terms
- Evidence-based, no filler phrases
- 600-800 words
- Write as an expert, not as an AI summarizing"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human",
     "Topic: {topic}\n\nResearch ({iterations} iteration(s), quality score: {score}/10):\n\n{content}"),
])

async def writer_node(state: ResearchState, api_key: str) -> dict:
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, google_api_key=api_key,
        temperature=0.6, max_output_tokens=1800,
    )
    chain = prompt | llm | StrOutputParser()

    report = await chain.ainvoke({
        "topic": state.topic,
        "iterations": state.iteration,
        "score": state.quality_score,
        "content": state.research_content,
    })

    tokens = len(report) // 4
    events = [
        {"agent": "writer", "type": "thought",
         "message": f"Synthesizing from {state.iteration} research pass(es), score {state.quality_score}/10"},
        {"agent": "writer", "type": "log",
         "message": "LCEL chain: PromptTemplate | Gemini | StrOutputParser"},
        {"agent": "writer", "type": "log",
         "message": f"Report complete — {len(report.split())} words, {state.total_tokens + tokens} total tokens"},
    ]

    return {
        "final_report": report,
        "total_tokens": state.total_tokens + tokens,
        "events": events,
    }
