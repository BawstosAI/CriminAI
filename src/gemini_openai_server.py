"""Gemini exposed as an OpenAI-compatible streaming API for vLLM integration."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import AsyncIterator, Optional

import dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

dotenv.load_dotenv()

try:
    from google import genai as google_genai
except ImportError as exc:
    raise ImportError("Install google-genai: pip install google-genai") from exc

app = FastAPI(title="Gemini OpenAI-Compatible API")

# Configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

_client: Optional[google_genai.Client] = None


def get_client() -> google_genai.Client:
    global _client
    if _client is None:
        _client = google_genai.Client(api_key=GOOGLE_API_KEY)
    return _client


# OpenAI-compatible request/response models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: list[ChatMessage]
    temperature: float = 1.0
    max_tokens: Optional[int] = None
    stream: bool = False
    top_p: Optional[float] = None
    stop: Optional[list[str]] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


def _convert_messages_to_gemini(messages: list[ChatMessage]) -> tuple[str, list[dict]]:
    """Convert OpenAI messages to Gemini format, extracting system prompt."""
    system_prompt = ""
    gemini_messages = []

    for msg in messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            gemini_messages.append({"role": "user", "parts": [{"text": msg.content}]})
        elif msg.role == "assistant":
            gemini_messages.append({"role": "model", "parts": [{"text": msg.content}]})

    return system_prompt, gemini_messages


async def _stream_gemini_response(
    request: ChatCompletionRequest,
) -> AsyncIterator[str]:
    """Stream response from Gemini in OpenAI SSE format."""
    client = get_client()
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    system_prompt, gemini_messages = _convert_messages_to_gemini(request.messages)

    # Build the model with system instruction
    model = client.models.get(name=f"models/{request.model}")

    # First chunk with role
    first_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[StreamChoice(delta=DeltaMessage(role="assistant", content=""))],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    try:
        # Use streaming chat
        response = client.models.generate_content_stream(
            model=f"models/{request.model}",
            contents=gemini_messages,
            config={
                "system_instruction": system_prompt if system_prompt else None,
                "temperature": request.temperature,
                "max_output_tokens": request.max_tokens,
                "top_p": request.top_p,
                "stop_sequences": request.stop,
            },
        )

        async def stream_chunks():
            # Run the synchronous generator in executor to not block
            loop = asyncio.get_running_loop()
            for chunk in response:
                if chunk.text:
                    stream_chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=request.model,
                        choices=[StreamChoice(delta=DeltaMessage(content=chunk.text))],
                    )
                    yield f"data: {stream_chunk.model_dump_json()}\n\n"
        
        async for chunk_data in stream_chunks():
            yield chunk_data

    except Exception as e:
        error_chunk = {"error": {"message": str(e), "type": "api_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"

    # Final chunk
    final_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible endpoint)."""
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    if request.stream:
        return StreamingResponse(
            _stream_gemini_response(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming response
    client = get_client()
    system_prompt, gemini_messages = _convert_messages_to_gemini(request.messages)

    try:
        response = client.models.generate_content(
            model=f"models/{request.model}",
            contents=gemini_messages,
            config={
                "system_instruction": system_prompt if system_prompt else None,
                "temperature": request.temperature,
                "max_output_tokens": request.max_tokens,
                "top_p": request.top_p,
                "stop_sequences": request.stop,
            },
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=response.text or ""),
                )
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "model": DEFAULT_MODEL}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("GEMINI_API_PORT", "8091"))
    uvicorn.run(app, host="0.0.0.0", port=port)
