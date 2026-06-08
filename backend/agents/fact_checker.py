"""
Fact-Checker Agent — LangChain LCEL chain with StrOutputParser
Evaluates the Search Agent's output for hallucinations and factual inaccuracies.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.core.models import ResearchState
from backend.core.config import GEMINI_MODEL

SYSTEM = """You are the Fact-Checker Agent in a LangGraph research pipeline.
Your job is to read the research content and identify any potential hallucinations, unsupported claims, or logical inconsistencies.
Be strict and thorough. If everything looks solidly supported by general knowledge, clearly state that the facts appear sound.
If there are issues, list them clearly so the Critic agent can reject the content and force a rewrite.

Keep your report concise (max 150 words)."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Topic: {topic}\n\nResearch content to verify:\n{content}"),
])

async def fact_checker_node(state: ResearchState, api_key: str) -> dict:
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL, google_api_key=api_key,
        temperature=0.1, max_output_tokens=300,
    )
    chain = prompt | llm | StrOutputParser()

    report = await chain.ainvoke({
        "topic": state.topic,
        "content": state.research_content,
    })

    # Estimate tokens
    tokens = len(report) // 4

    events = [
        {"agent": "fact_checker", "type": "thought",
         "message": f"Verifying facts for pass {state.iteration}..."},
        {"agent": "fact_checker", "type": "log",
         "message": "Analyzed content for hallucinations and unsupported claims"},
        {"agent": "fact_checker", "type": "log",
         "message": f"Fact-Check Report: {report[:100]}..."},
    ]

    return {
        "fact_check_report": report,
        "total_tokens": state.total_tokens + tokens + 100,
        "events": events,
    }
