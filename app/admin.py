from typing import List, Optional

import bcrypt
import os
from fastapi import APIRouter, Depends, Header, HTTPException, status
import mysql.connector
from mysql.connector import errorcode
from pydantic import BaseModel, EmailStr, Field

from app.db_core import get_core_connection

router = APIRouter()


def _require_admin(api_key: str = Header(..., alias="X-Admin-Key")) -> None:
    expected_key = os.getenv("ADMIN_API_KEY")
    if not expected_key or api_key != expected_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin credentials")


def _has_column(cursor, table_name: str, column_name: str) -> bool:
    """Return True when the given column exists on the target table."""

    try:
        cursor.execute(f"SHOW COLUMNS FROM {table_name} LIKE %s", (column_name,))
        return cursor.fetchone() is not None
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_NO_SUCH_TABLE:
            return False
        raise


class StoreAssignment(BaseModel):
    store_db: str = Field(..., description="Database name for the tenant store")
    db_user: str = Field(..., description="Database user with read access")
    db_pass: str = Field(..., description="Database password for the store user")
    store_name: Optional[str] = Field(None, description="Friendly name for the store")


class StoreAssignmentRecord(StoreAssignment):
    id: Optional[int] = Field(None, description="Identifier from the user_stores table when available")


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


class AdminUser(BaseModel):
    id: int
    email: EmailStr
    full_name: Optional[str]
    stores: List[StoreAssignmentRecord] = Field(default_factory=list)


class AdminUsersResponse(BaseModel):
    users: List[AdminUser]


@router.post("/users", response_model=CreateUserResponse, status_code=status.HTTP_201_CREATED)
def create_user(payload: CreateUserRequest, _: None = Depends(_require_admin)):
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        supports_full_name = _has_column(cursor, "users", "full_name")

        cursor.execute("SELECT id FROM users WHERE email = %s", (payload.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User already exists")

        password_hash = bcrypt.hashpw(payload.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        insert_columns = ["email", "password_hash"]
        insert_values = [payload.email, password_hash]

        stored_full_name: Optional[str] = None
        if payload.full_name and supports_full_name:
            insert_columns.append("full_name")
            insert_values.append(payload.full_name)
            stored_full_name = payload.full_name

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
            full_name=stored_full_name,
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


def _ensure_user_exists(cursor, user_id: int) -> None:
    cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
    if not cursor.fetchone():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")


def _list_store_assignments(cursor, user_id: int) -> List[StoreAssignmentRecord]:
    stores: List[StoreAssignmentRecord] = []

    try:
        cursor.execute(
            """
            SELECT id, store_db, db_user, db_pass, store_name
            FROM user_stores
            WHERE user_id = %s
            ORDER BY id
            """,
            (user_id,),
        )
        for row in cursor.fetchall():
            stores.append(
                StoreAssignmentRecord(
                    id=row.get("id"),
                    store_db=row.get("store_db"),
                    db_user=row.get("db_user"),
                    db_pass=row.get("db_pass"),
                    store_name=row.get("store_name"),
                )
            )
    except mysql.connector.Error as err:
        if err.errno != errorcode.ER_NO_SUCH_TABLE:
            raise

    if not stores:
        cursor.execute(
            "SELECT store_db, db_user, db_pass, store_name FROM users WHERE id = %s",
            (user_id,),
        )
        row = cursor.fetchone()
        if row and row.get("store_db"):
            stores.append(
                StoreAssignmentRecord(
                    store_db=row.get("store_db"),
                    db_user=row.get("db_user"),
                    db_pass=row.get("db_pass"),
                    store_name=row.get("store_name"),
                )
            )

    return stores


@router.get("/users", response_model=AdminUsersResponse)
def list_users(_: None = Depends(_require_admin)):
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        supports_full_name = _has_column(cursor, "users", "full_name")

        selected_columns = ["id", "email"]
        if supports_full_name:
            selected_columns.append("full_name")

        column_clause = ", ".join(selected_columns)
        cursor.execute(f"SELECT {column_clause} FROM users ORDER BY email")
        rows = cursor.fetchall() or []

        users: List[AdminUser] = []
        for row in rows:
            stores = _list_store_assignments(cursor, row["id"])
            users.append(
                AdminUser(
                    id=row["id"],
                    email=row["email"],
                    full_name=row.get("full_name"),
                    stores=stores,
                )
            )

        return AdminUsersResponse(users=users)
    except HTTPException:
        raise
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Failed to load users: {err}")
    finally:
        cursor.close()
        conn.close()


@router.get("/users/{user_id}", response_model=AdminUser)
def get_user(user_id: int, _: None = Depends(_require_admin)):
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        supports_full_name = _has_column(cursor, "users", "full_name")

        selected_columns = ["id", "email"]
        if supports_full_name:
            selected_columns.append("full_name")

        column_clause = ", ".join(selected_columns)
        cursor.execute(
            f"SELECT {column_clause} FROM users WHERE id = %s",
            (user_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        stores = _list_store_assignments(cursor, user_id)

        return AdminUser(
            id=row["id"],
            email=row["email"],
            full_name=row.get("full_name"),
            stores=stores,
        )
    except HTTPException:
        raise
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Failed to load user: {err}")
    finally:
        cursor.close()
        conn.close()


class StoreAssignmentResponse(BaseModel):
    store: StoreAssignmentRecord


@router.post(
    "/users/{user_id}/stores",
    response_model=StoreAssignmentResponse,
    status_code=status.HTTP_201_CREATED,
)
def add_store_to_user(user_id: int, payload: StoreAssignment, _: None = Depends(_require_admin)):
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        _ensure_user_exists(cursor, user_id)

        try:
            cursor.execute(
                "SELECT id FROM user_stores WHERE user_id = %s AND store_db = %s",
                (user_id, payload.store_db),
            )
        except mysql.connector.Error as err:
            if err.errno != errorcode.ER_NO_SUCH_TABLE:
                raise
        else:
            if cursor.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Store already assigned to this user",
                )

        record_id: Optional[int] = None

        try:
            cursor.execute(
                """
                INSERT INTO user_stores (user_id, store_db, db_user, db_pass, store_name)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    payload.store_db,
                    payload.db_user,
                    payload.db_pass,
                    payload.store_name,
                ),
            )
            record_id = cursor.lastrowid
        except mysql.connector.Error as err:
            if err.errno != errorcode.ER_NO_SUCH_TABLE:
                raise

            cursor.execute(
                "SELECT store_db FROM users WHERE id = %s",
                (user_id,),
            )
            existing = cursor.fetchone()
            if existing and existing.get("store_db"):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="User already has a store assignment",
                )

            cursor.execute(
                """
                UPDATE users
                SET store_db = %s, db_user = %s, db_pass = %s, store_name = %s
                WHERE id = %s
                """,
                (
                    payload.store_db,
                    payload.db_user,
                    payload.db_pass,
                    payload.store_name,
                    user_id,
                ),
            )

        conn.commit()

        return StoreAssignmentResponse(
            store=StoreAssignmentRecord(id=record_id, **payload.dict()),
        )
    except HTTPException:
        conn.rollback()
        raise
    except mysql.connector.Error as err:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to assign store: {err}")
    finally:
        cursor.close()
        conn.close()


class RemoveStoreResponse(BaseModel):
    success: bool
    stores: List[StoreAssignmentRecord]


@router.delete("/users/{user_id}/stores/{store_identifier}", response_model=RemoveStoreResponse)
def remove_store_from_user(user_id: int, store_identifier: str, _: None = Depends(_require_admin)):
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        _ensure_user_exists(cursor, user_id)

        deleted = False

        try:
            try:
                store_id = int(store_identifier)
                cursor.execute(
                    "DELETE FROM user_stores WHERE user_id = %s AND id = %s",
                    (user_id, store_id),
                )
            except ValueError:
                cursor.execute(
                    "DELETE FROM user_stores WHERE user_id = %s AND store_db = %s",
                    (user_id, store_identifier),
                )

            if cursor.rowcount:
                deleted = True
                conn.commit()
        except mysql.connector.Error as err:
            if err.errno != errorcode.ER_NO_SUCH_TABLE:
                raise

            cursor.execute(
                "SELECT store_db FROM users WHERE id = %s",
                (user_id,),
            )
            row = cursor.fetchone()
            current_db = row.get("store_db") if row else None
            if not current_db:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Store assignment not found for user",
                )

            if store_identifier not in ("legacy", current_db):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Store assignment not found for user",
                )

            cursor.execute(
                """
                UPDATE users
                SET store_db = NULL, db_user = NULL, db_pass = NULL, store_name = NULL
                WHERE id = %s
                """,
                (user_id,),
            )
            conn.commit()
            deleted = True

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Store assignment not found for user",
            )

        stores = _list_store_assignments(cursor, user_id)

        return RemoveStoreResponse(success=True, stores=stores)
    except HTTPException:
        conn.rollback()
        raise
    except mysql.connector.Error as err:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to remove store: {err}")
    finally:
        cursor.close()
        conn.close()
