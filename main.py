from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from backend.api.routes import router

app = FastAPI(title="Research Orchestrator", version="2.0.0",
              description="LangGraph + LangChain multi-agent research system")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(router)

frontend_path = os.path.join(os.path.dirname(__file__), "frontend")

@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(os.path.join(frontend_path, "index.html"))

app.mount("/css", StaticFiles(directory=os.path.join(frontend_path,"css")), name="css")
app.mount("/js",  StaticFiles(directory=os.path.join(frontend_path,"js")),  name="js")
