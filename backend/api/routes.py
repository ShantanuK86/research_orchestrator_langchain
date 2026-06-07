"""
routes.py
─────────
FastAPI SSE endpoint.
Runs the LangGraph graph and streams AgentEvents to the frontend.
"""
import json
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from backend.core.models import ResearchRequest, AgentEvent, ResearchState
from backend.graph.research_graph import build_graph
from backend.core.config import GEMINI_API_KEY

router = APIRouter(prefix="/api/v1", tags=["orchestration"])

NODE_AGENT_MAP = {
    "supervisor": "supervisor",
    "search":     "search",
    "critic":     "critic",
    "writer":     "writer",
}

def sse(event: AgentEvent) -> str:
    return f"data: {json.dumps(event.model_dump())}\n\n"

def sse_raw(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def stream_graph(req: ResearchRequest):
    api_key = req.api_key or GEMINI_API_KEY
    start = time.time()

    # Signal pipeline start
    yield sse(AgentEvent(agent="system", type="status", message="started"))
    yield sse(AgentEvent(agent="supervisor", type="status", message="active"))
    yield sse(AgentEvent(agent="supervisor", type="log",
                         message=f'Query received: "{req.query}"'))

    try:
        graph = build_graph(api_key)
        init_state = ResearchState(
            query=req.query,
            max_iterations=req.max_iterations or 3,
        )

        last_node = None

        # Stream node-by-node via LangGraph's astream
        async for step in graph.astream(init_state, stream_mode="updates"):
            for node_name, node_output in step.items():
                agent = NODE_AGENT_MAP.get(node_name, "system")

                # Transition: mark previous node done, activate new one
                if last_node and last_node != node_name:
                    yield sse(AgentEvent(agent=NODE_AGENT_MAP.get(last_node, "system"),
                                         type="status", message="done"))
                    yield sse(AgentEvent(agent="supervisor", type="status", message="active"))
                    yield sse(AgentEvent(agent="supervisor", type="log",
                                         message=f"Routing: {last_node} → {node_name}"))
                    yield sse(AgentEvent(agent="supervisor", type="status", message="done"))

                yield sse(AgentEvent(agent=agent, type="status", message="active"))

                # Drain the event queue this node produced
                for evt in node_output.get("events", []):
                    yield sse(AgentEvent(
                        agent=evt.get("agent", agent),
                        type=evt.get("type", "log"),
                        message=evt.get("message", ""),
                        thinking=evt.get("thinking", False),
                        data=evt.get("data"),
                    ))

                # Emit metrics after each node
                iteration = node_output.get("iteration", 0)
                tokens    = node_output.get("total_tokens", 0)
                if iteration:
                    yield sse(AgentEvent(agent="system", type="metric", message="",
                                         data={"iteration": iteration}))
                if tokens:
                    yield sse(AgentEvent(agent="system", type="metric", message="",
                                         data={"tokens": tokens}))

                # Emit critic score for the UI score ring
                if node_name == "critic":
                    score   = node_output.get("quality_score", 0)
                    approve = node_output.get("approved", False)
                    yield sse(AgentEvent(agent="critic", type="score", message="",
                                         data={"score": score, "approve": approve,
                                               "iteration": node_output.get("iteration", 1)}))

                last_node = node_name

        # Mark writer done and emit final report
        if last_node:
            yield sse(AgentEvent(agent=NODE_AGENT_MAP.get(last_node, "system"),
                                  type="status", message="done"))

        # Get final state
        final = await graph.ainvoke(init_state) if False else None  # already ran above

    except Exception as e:
        yield sse(AgentEvent(agent="system", type="log", message=f"Graph error: {e}"))
        yield "data: [DONE]\n\n"
        return

    # We need to re-run to get final state — instead collect it from stream
    # Re-run cleanly to get final_report (LangGraph streams don't expose final state easily)
    try:
        graph2 = build_graph(api_key)
        final_state = await graph2.ainvoke(ResearchState(
            query=req.query,
            max_iterations=req.max_iterations or 3,
        ))
    except Exception as e:
        yield sse(AgentEvent(agent="system", type="log", message=f"Final state error: {e}"))
        yield "data: [DONE]\n\n"
        return

    elapsed = round(time.time() - start, 1)
    yield sse(AgentEvent(agent="supervisor", type="log",
                          message=f"✓ Pipeline complete — {final_state.iteration} iteration(s) · "
                                  f"{final_state.total_tokens} tokens · {elapsed}s"))
    yield sse(AgentEvent(agent="supervisor", type="status", message="done",
                          data={"label": "complete"}))
    yield sse(AgentEvent(agent="system", type="metric", message="",
                          data={"tokens": final_state.total_tokens, "elapsed": elapsed}))
    yield sse(AgentEvent(agent="writer", type="result", message="",
                          data={
                              "topic":           final_state.topic,
                              "report":          final_state.final_report,
                              "iterations":      final_state.iteration,
                              "total_tokens":    final_state.total_tokens,
                              "elapsed_seconds": elapsed,
                              "quality_score":   final_state.quality_score,
                          }))
    yield "data: [DONE]\n\n"


@router.post("/research/stream")
async def research_stream(req: ResearchRequest):
    if not (req.api_key or GEMINI_API_KEY):
        raise HTTPException(status_code=400, detail="Gemini API key required.")
    return StreamingResponse(
        stream_graph(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@router.get("/health")
async def health():
    return {"status": "ok", "service": "research-orchestrator", "framework": "LangGraph 0.4"}
