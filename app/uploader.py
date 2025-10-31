from typing import Optional, Dict, Any

import os
from fastapi import APIRouter, HTTPException, Header, Query, status
from pydantic import BaseModel

from app.db_core import get_core_connection


router = APIRouter()


class CredsResponse(BaseModel):
    engine: str = "mysql"
    host: str
    port: int = 3306
    database: str
    username: str
    password: str
    schema: Optional[str] = None


def _load_store_row(store_id: Optional[int], store_db: Optional[str]) -> Optional[Dict[str, Any]]:
    """Fetch a store assignment row from user_stores or users (legacy fallback).

    Accepts either numeric store_id or a store_db name. Returns a dict with
    store_db, db_user, db_pass, and api_key if present.
    """
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        if store_id is not None:
            try:
                cursor.execute(
                    "SELECT store_db, db_user, db_pass, api_key FROM user_stores WHERE id = %s",
                    (store_id,),
                )
                row = cursor.fetchone()
                if row:
                    return row
            except Exception:
                pass

        if store_db:
            try:
                cursor.execute(
                    "SELECT store_db, db_user, db_pass, api_key FROM user_stores WHERE store_db = %s",
                    (store_db,),
                )
                row = cursor.fetchone()
                if row:
                    return row
            except Exception:
                pass

        # Legacy: fallback to users table single-store columns (no api_key)
        if store_db:
            cursor.execute(
                "SELECT store_db, db_user, db_pass FROM users WHERE store_db = %s",
                (store_db,),
            )
            row = cursor.fetchone()
            return row
        return None
    finally:
        try:
            cursor.close(); conn.close()
        except Exception:
            pass


@router.get("/creds", response_model=CredsResponse)
def get_uploader_creds(
    x_api_key: str = Header(..., alias="X-API-Key"),
    store_id: Optional[int] = Query(None, description="ID from user_stores"),
    store_db: Optional[str] = Query(None, description="Database name for the store (tenant DB)"),
):
    """Return DB credentials for the uploader.

    Auth:
      - Primary: per-store API key stored in user_stores.api_key
      - Fallback (legacy): global ADMIN_API_KEY (use only until api_key exists)
    Host/Port are provided via env: STORE_RDS_HOST, STORE_RDS_PORT (default 3306)
    """
    if not store_id and not store_db:
        raise HTTPException(status_code=400, detail="Provide store_id or store_db")

    row = _load_store_row(store_id, store_db)
    if not row:
        raise HTTPException(status_code=404, detail="Store not found")

    # AuthZ: prefer per-store key, fallback to admin key to ease migration
    per_store_key = row.get("api_key") if isinstance(row, dict) else None
    admin_key = os.getenv("ADMIN_API_KEY")

    if per_store_key:
        if x_api_key != per_store_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    else:
        # Fallback only if global key is configured
        if not admin_key or x_api_key != admin_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    host = os.getenv("STORE_RDS_HOST")
    if not host:
        raise HTTPException(status_code=500, detail="STORE_RDS_HOST not configured")
    try:
        port = int(os.getenv("STORE_RDS_PORT", "3306"))
    except ValueError:
        port = 3306

    db_name = row.get("store_db")
    db_user = row.get("db_user")
    db_pass = row.get("db_pass")
    if not all([db_name, db_user, db_pass]):
        raise HTTPException(status_code=500, detail="Incomplete store credentials")

    return CredsResponse(
        engine="mysql",
        host=host,
        port=port,
        database=db_name,
        username=db_user,
        password=db_pass,
        schema=None,
    )


