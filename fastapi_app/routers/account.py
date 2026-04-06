from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, Request

from fastapi_app.dependencies import AuthUser, get_current_user, get_optional_user
from fastapi_app.services.billing_service import BillingService

router = APIRouter(prefix="/account", tags=["account"])


def get_service() -> BillingService:
    return BillingService()


@router.get("/runtime-config")
async def get_runtime_config(service: BillingService = Depends(get_service)) -> Dict[str, Any]:
    return service.get_runtime_config()


@router.get("/quota")
async def get_quota(
    request: Request,
    user: Optional[AuthUser] = Depends(get_optional_user),
    service: BillingService = Depends(get_service),
) -> Dict[str, Any]:
    guest_id = getattr(request.state, "guest_id", None)
    return service.get_quota(user=user, guest_id=guest_id)


@router.post("/quota/consume")
async def consume_quota(
    request: Request,
    workflow_type: str = Body(..., embed=True),
    amount: Optional[int] = Body(None, embed=True),
    user: Optional[AuthUser] = Depends(get_optional_user),
    service: BillingService = Depends(get_service),
) -> Dict[str, Any]:
    guest_id = getattr(request.state, "guest_id", None)
    return service.consume_workflow(
        workflow_type=workflow_type,
        amount=amount,
        user=user,
        guest_id=guest_id,
    )


@router.get("/profile")
async def get_profile(
    user: AuthUser = Depends(get_current_user),
    service: BillingService = Depends(get_service),
) -> Dict[str, Any]:
    return service.get_account_profile(user)


@router.post("/invite/claim")
async def claim_invite_code(
    invite_code: str = Body(..., embed=True),
    user: AuthUser = Depends(get_current_user),
    service: BillingService = Depends(get_service),
) -> Dict[str, Any]:
    return service.claim_invite_code(user=user, invite_code=invite_code)


@router.post("/points/redeem")
async def redeem_points_code(
    redeem_code: str = Body(..., embed=True),
    user: AuthUser = Depends(get_current_user),
    service: BillingService = Depends(get_service),
) -> Dict[str, Any]:
    return service.redeem_points_code(user=user, redeem_code=redeem_code)
