import os
import time
import traceback
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from rag_engine import EnterpriseRAG

app = FastAPI(title="Nexus AI", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------------------------------------------------------
# SESSION MANAGEMENT
# Each user/chat gets their own AI engine instance (separate memory).
# sessions = { session_id: { "engine": EnterpriseRAG, "last_used": timestamp } }
# ------------------------------------------------------------------
sessions: dict = {}
SESSION_TIMEOUT = 1800  # 30 minutes of inactivity → session cleared


def get_or_create_session(session_id: str) -> EnterpriseRAG:
    """Get existing AI engine for session or create a new one."""
    now = time.time()

    # Purge expired sessions to prevent memory leak
    expired = [sid for sid, s in sessions.items() if now - s["last_used"] > SESSION_TIMEOUT]
    for sid in expired:
        del sessions[sid]

    if session_id not in sessions:
        sessions[session_id] = {"engine": EnterpriseRAG(), "last_used": now}
    else:
        sessions[session_id]["last_used"] = now

    return sessions[session_id]["engine"]


# ------------------------------------------------------------------
# STARTUP
# ------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    print("[NEXUS AI] Server starting up...")
    print(f"[NEXUS AI] GROQ_API_KEY present: {bool(os.environ.get('GROQ_API_KEY'))}")


# ------------------------------------------------------------------
# DATA MODELS
# ------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"


# ------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


@app.post("/api/reset")
async def reset_session(request: ChatRequest):
    """Clear conversation memory for a given session (New Chat button)."""
    sid = request.session_id or "default"
    if sid in sessions:
        sessions[sid]["engine"].reset()
    return {"success": True, "message": "Session reset."}


@app.get("/api/new_session")
async def new_session():
    """Generate a fresh unique session ID for a new conversation."""
    return {"session_id": str(uuid.uuid4())}


@app.post("/api/stream")
async def stream_chat(request: ChatRequest):
    """
    Streaming endpoint using POST to allow large file contents.
    Streams AI response token-by-token.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    engine = get_or_create_session(request.session_id or "default")

    async def event_generator():
        try:
            async for chunk in engine.astream_question(request.question):
                yield str(chunk)
        except Exception as e:
            print("ERROR IN STREAM:")
            traceback.print_exc()
            yield f"\n\n[ERROR] {str(e)}"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    }
    return StreamingResponse(event_generator(), media_type="text/plain", headers=headers)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint (fallback)."""
    engine = get_or_create_session(request.session_id or "default")
    try:
        result = await engine.ask_question(request.question)
        return {"success": True, "answer": result["answer"]}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
