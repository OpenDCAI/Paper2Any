from __future__ import annotations

from fastapi import APIRouter, Body, Depends
from fastapi_app.schemas import VerifyLlmRequest, VerifyLlmResponse
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

router = APIRouter()


def get_service() -> Paper2AnyService:
    from fastapi_app.services.paper2any_service import Paper2AnyService

    return Paper2AnyService()


@router.post("/system/verify-llm", response_model=VerifyLlmResponse)
async def verify_llm_connection(
    req: VerifyLlmRequest = Body(...),
    service: Paper2AnyService = Depends(get_service),
):
    """
    Verify LLM connection by sending a simple 'Hi' message from the backend.
    """
    return await service.verify_llm_connection(req)
