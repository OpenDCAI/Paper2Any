"""
Mock OpenAI-compatible API server for testing.

This server simulates LLM and image generation APIs, allowing tests to run
without actual API keys or model deployments.

Usage:
    python -m tests.mocks.mock_api_server

    Or in tests:
    from tests.mocks.mock_api_server import app
    # Use with TestClient or run as subprocess
"""

import base64
import json
import time
import uuid
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Mock OpenAI API Server")


# ============================================================================
# Request/Response Models
# ============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = "dall-e-3"
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"


# ============================================================================
# Mock Response Generators
# ============================================================================

MOCK_RESPONSES = {
    # General responses for different types of prompts
    "default": "This is a mock response from the test server. The actual content would come from the real LLM API.",
    "json": '{"result": "mock_data", "status": "success"}',
    "code": "```python\ndef mock_function():\n    return 'Hello from mock!'\n```",
    "analysis": "Based on the analysis, the document contains: 1) Introduction section, 2) Methods section, 3) Results section, 4) Conclusion. This is mock analysis data.",
}


def generate_mock_response(messages: list[dict]) -> str:
    """Generate appropriate mock response based on the prompt."""
    if not messages:
        return MOCK_RESPONSES["default"]

    last_message = messages[-1].get("content", "").lower()

    # Detect prompt type and return appropriate mock
    if "json" in last_message or "格式" in last_message:
        return MOCK_RESPONSES["json"]
    elif "code" in last_message or "代码" in last_message:
        return MOCK_RESPONSES["code"]
    elif "analy" in last_message or "分析" in last_message:
        return MOCK_RESPONSES["analysis"]

    return MOCK_RESPONSES["default"]


def create_chat_completion_response(request: ChatCompletionRequest) -> dict:
    """Create a mock chat completion response."""
    response_id = f"chatcmpl-mock-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    content = generate_mock_response(request.messages)

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 100,
            "total_tokens": 150,
        },
    }


def create_streaming_response(request: ChatCompletionRequest):
    """Generate SSE streaming response."""
    response_id = f"chatcmpl-mock-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    content = generate_mock_response(request.messages)

    # Split content into chunks for streaming
    chunks = [content[i:i+10] for i in range(0, len(content), 10)]

    def generate():
        for i, chunk in enumerate(chunks):
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk} if i > 0 else {"role": "assistant", "content": chunk},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.05)  # Simulate streaming delay

        # Final chunk
        final = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return generate()


# ============================================================================
# Mock Image Generation
# ============================================================================

# 1x1 transparent PNG as base64
MOCK_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


def create_image_response(request: ImageGenerationRequest) -> dict:
    """Create a mock image generation response."""
    created = int(time.time())

    images = []
    for _ in range(request.n):
        if request.response_format == "b64_json":
            images.append({"b64_json": MOCK_IMAGE_B64})
        else:
            # Return a placeholder URL (would need actual hosting in real scenario)
            images.append({"url": f"https://mock-images.test/generated-{uuid.uuid4().hex[:8]}.png"})

    return {
        "created": created,
        "data": images,
    }


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "server": "mock-openai-api"}


@app.get("/v1/models")
async def list_models():
    """List available models (mock)."""
    return {
        "object": "list",
        "data": [
            {"id": "gpt-4", "object": "model", "owned_by": "mock"},
            {"id": "gpt-3.5-turbo", "object": "model", "owned_by": "mock"},
            {"id": "dall-e-3", "object": "model", "owned_by": "mock"},
        ],
    }


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Mock chat completion endpoint."""
    if request.stream:
        return StreamingResponse(
            create_streaming_response(request),
            media_type="text/event-stream",
        )
    return JSONResponse(create_chat_completion_response(request))


@app.post("/v1/images/generations")
@app.post("/images/generations")
async def image_generations(request: ImageGenerationRequest):
    """Mock image generation endpoint."""
    return JSONResponse(create_image_response(request))


@app.post("/v1/images/edits")
@app.post("/images/edits")
async def image_edits(request: Request):
    """Mock image edit endpoint."""
    # For multipart form data
    return JSONResponse({
        "created": int(time.time()),
        "data": [{"b64_json": MOCK_IMAGE_B64}],
    })


# ============================================================================
# Vision API (for image understanding)
# ============================================================================

@app.post("/v1/chat/completions/vision")
async def vision_completions(request: Request):
    """Mock vision completion endpoint for image understanding."""
    body = await request.json()

    return JSONResponse({
        "id": f"chatcmpl-vision-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "gpt-4-vision-preview"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock image analysis. The image appears to contain visual elements that would be analyzed by the actual vision model.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    })


# ============================================================================
# Run server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting Mock OpenAI API Server on http://localhost:8080")
    print("Endpoints:")
    print("  POST /v1/chat/completions - Chat completions")
    print("  POST /v1/images/generations - Image generation")
    print("  GET  /v1/models - List models")
    print("  GET  /health - Health check")
    uvicorn.run(app, host="0.0.0.0", port=8080)
