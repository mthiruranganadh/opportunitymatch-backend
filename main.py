import json
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import BaseModel, Field

from agents_setup import supervisor_agent

load_dotenv()

APP_NAME = "opportunitymatch_ai"
runner = InMemoryRunner(agent=supervisor_agent, app_name=APP_NAME)

app = FastAPI(title="OpportunityMatch AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "https://opportunitymatch-frontend-mthiruranganadh.vercel.app/",
        "https://opportunitymatch-frontend-mthiruranganadh.vercel.app",
        "https://opportunitymatch-frontend.vercel.app/",
        "https://opportunitymatch-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    profile: dict[str, Any] = Field(
        default_factory=dict,
        description="Student details such as education, skills, interests, and goals."
    )
    message: str = "Find the best scholarships and internships for this profile."
    user_id: str = "student"
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


async def _get_or_create_session(user_id: str, session_id: str):
    existing_session = await runner.session_service.get_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )
    if existing_session:
        return existing_session

    return await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or uuid4().hex
    await _get_or_create_session(request.user_id, session_id)

    prompt = (
        f"Student profile:\n{json.dumps(request.profile, indent=2)}\n\n"
        f"Student request: {request.message}"
    )
    user_message = types.Content(
        role="user",
        parts=[types.Part(text=prompt)],
    )

    final_response = ""
    try:
        async for event in runner.run_async(
            user_id=request.user_id,
            session_id=session_id,
            new_message=user_message,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_response = "".join(
                    part.text for part in event.content.parts if part.text
                )
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Unable to generate opportunity matches: {error}",
        ) from error

    if not final_response:
        raise HTTPException(
            status_code=502,
            detail="The agent did not return a final response.",
        )

    return ChatResponse(response=final_response, session_id=session_id)
