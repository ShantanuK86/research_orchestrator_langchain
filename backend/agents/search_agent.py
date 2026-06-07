"""
Search Agent — LangChain LCEL chain with MessagesPlaceholder for context
Uses HumanMessage + SystemMessage from langchain_core.messages
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from backend.core.models import ResearchState
from backend.core.config import GEMINI_MODEL

SYSTEM = """You are the Search Agent in a LangGraph research pipeline.
Synthesize a comprehensive knowledge base on the given topic.

Cover:
- Key facts, definitions, background context
- Current state and recent developments
- Expert perspectives, debates, and conflicting views
- Statistics, benchmarks, comparisons
- Real-world use cases and examples
- Future implications

Use clear markdown sections. Be thorough and accurate. Target 500-700 words."""

async def search_node(state: ResearchState, api_key: str) -> dict:
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, google_api_key=api_key,
        temperature=0.7, max_output_tokens=1500,
    )

    # Build context-aware prompt — on retry, focus on gaps
    if state.iteration == 1:
        queries_fmt = "\n".join(f"{i+1}. {q}" for i, q in enumerate(state.search_queries))
        human_msg = (
            f"Topic: {state.topic}\n\n"
            f"Research strategy: {state.plan}\n\n"
            f"Cover these angles:\n{queries_fmt}"
        )
    else:
        gaps_fmt = "\n".join(f"- {g}" for g in state.gaps) if state.gaps else "- Improve overall depth"
        human_msg = (
            f"Topic: {state.topic}\n\n"
            f"Refinement pass #{state.iteration}. Critic flagged these gaps:\n{gaps_fmt}\n\n"
            f"Address these specifically while maintaining full topic coverage."
        )

    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=human_msg)]

    # Direct LLM invoke with message list (LangChain message format)
    response = await llm.ainvoke(messages)
    text = response.content
    tokens = getattr(response, "usage_metadata", {})
    token_count = tokens.get("output_tokens", len(text) // 4) if isinstance(tokens, dict) else len(text) // 4

    word_count = len(text.split())
    events = [
        {"agent": "search", "type": "thought",
         "message": f"Pass {state.iteration} — querying {len(state.search_queries)} angles"},
        {"agent": "search", "type": "log",
         "message": f"Using LangChain HumanMessage + SystemMessage → Gemini"},
        {"agent": "search", "type": "log",
         "message": f"Gathered {word_count} words across {len(state.search_queries)} research angles"},
    ]

    if state.iteration > 1 and state.gaps:
        events.append({"agent": "search", "type": "log",
            "message": f"Addressed {len(state.gaps)} gap(s) from Critic: {'; '.join(state.gaps[:2])}"})

    return {
        "research_content": text,
        "total_tokens": state.total_tokens + token_count,
        "events": events,
    }
