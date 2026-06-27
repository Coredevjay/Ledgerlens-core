"""Wallet allowlist/denylist management API (Issue #181)."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.auth import require_admin_key
from detection.wallet_override_store import (
    add_override,
    list_overrides,
    remove_override,
)

router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin_key)], tags=["Allowlist / Denylist"])


class OverrideCreate(BaseModel):
    wallet: str
    reason: str
    added_by: str


@router.post(
    "/allowlist",
    status_code=201,
    summary="Add wallet to allowlist",
    description="Flag a wallet as trusted. Subsequent score lookups return score=0 with override='allowlisted'.",
)
def add_to_allowlist(body: OverrideCreate) -> dict:
    try:
        return add_override(body.wallet, "allowlist", body.reason, body.added_by)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post(
    "/denylist",
    status_code=201,
    summary="Add wallet to denylist",
    description="Flag a wallet as a confirmed bad actor. Score lookups return score=100 with override='denylisted'.",
)
def add_to_denylist(body: OverrideCreate) -> dict:
    try:
        return add_override(body.wallet, "denylist", body.reason, body.added_by)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get(
    "/allowlist",
    summary="List allowlist entries",
    description="Return paginated allowlist entries including soft-deleted (removed_at non-null) rows.",
)
def get_allowlist(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[dict]:
    return list_overrides("allowlist", limit=limit, offset=offset)


@router.get(
    "/denylist",
    summary="List denylist entries",
    description="Return paginated denylist entries including soft-deleted rows.",
)
def get_denylist(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[dict]:
    return list_overrides("denylist", limit=limit, offset=offset)


@router.delete(
    "/allowlist/{wallet}",
    summary="Remove wallet from allowlist",
    description="Soft-delete: preserves the audit trail row with removed_at timestamp.",
)
def delete_from_allowlist(
    wallet: str,
    removed_by: str = Query(..., description="Actor performing the removal"),
) -> dict:
    result = remove_override(wallet, "allowlist", removed_by)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Wallet {wallet} not found on allowlist")
    return result


@router.delete(
    "/denylist/{wallet}",
    summary="Remove wallet from denylist",
    description="Soft-delete: preserves the audit trail row with removed_at timestamp.",
)
def delete_from_denylist(
    wallet: str,
    removed_by: str = Query(..., description="Actor performing the removal"),
) -> dict:
    result = remove_override(wallet, "denylist", removed_by)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Wallet {wallet} not found on denylist")
    return result
