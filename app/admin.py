from typing import List, Optional

import bcrypt
import os
from fastapi import APIRouter, Depends, Header, HTTPException, status
import mysql.connector
from pydantic import BaseModel, EmailStr, Field

from app.db_core import get_core_connection

router = APIRouter()


def _require_admin(api_key: str = Header(..., alias="X-Admin-Key")) -> None:
    expected_key = os.getenv("ADMIN_API_KEY")
    if not expected_key or api_key != expected_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin credentials")


class StoreAssignment(BaseModel):
    store_db: str = Field(..., description="Database name for the tenant store")
    db_user: str = Field(..., description="Database user with read access")
    db_pass: str = Field(..., description="Database password for the store user")
    store_name: Optional[str] = Field(None, description="Friendly name for the store")


class CreateUserRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = Field(None, description="Optional display name for the user")
    stores: List[StoreAssignment] = Field(default_factory=list)


class CreateUserResponse(BaseModel):
    id: int
    email: EmailStr
    full_name: Optional[str]
    stores: List[StoreAssignment]


@router.post("/users", response_model=CreateUserResponse, status_code=status.HTTP_201_CREATED)
def create_user(payload: CreateUserRequest, _: None = Depends(_require_admin)):
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("SELECT id FROM users WHERE email = %s", (payload.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User already exists")

        password_hash = bcrypt.hashpw(payload.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        insert_columns = ["email", "password_hash"]
        insert_values = [payload.email, password_hash]

        if payload.full_name:
            insert_columns.append("full_name")
            insert_values.append(payload.full_name)

        placeholders = ", ".join(["%s"] * len(insert_values))
        column_clause = ", ".join(insert_columns)

        cursor.execute(
            f"INSERT INTO users ({column_clause}) VALUES ({placeholders})",
            tuple(insert_values),
        )
        user_id = cursor.lastrowid

        legacy_store_written = False
        for store in payload.stores:
            try:
                cursor.execute(
                    """
                    INSERT INTO user_stores (user_id, store_db, db_user, db_pass, store_name)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        user_id,
                        store.store_db,
                        store.db_user,
                        store.db_pass,
                        store.store_name,
                    ),
                )
            except mysql.connector.Error:
                if legacy_store_written or len(payload.stores) != 1:
                    raise

                cursor.execute(
                    """
                    UPDATE users
                    SET store_db = %s, db_user = %s, db_pass = %s, store_name = %s
                    WHERE id = %s
                    """,
                    (
                        store.store_db,
                        store.db_user,
                        store.db_pass,
                        store.store_name,
                        user_id,
                    ),
                )
                legacy_store_written = True

        conn.commit()

        return CreateUserResponse(
            id=user_id,
            email=payload.email,
            full_name=payload.full_name,
            stores=payload.stores,
        )
    except HTTPException:
        conn.rollback()
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create user: {exc}")
    finally:
        cursor.close()
        conn.close()
