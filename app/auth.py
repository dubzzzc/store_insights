from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import jwt, os, bcrypt
from typing import List, Dict, Any

from app.db_core import get_core_connection
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

print("✅ auth.py loaded and router defined")


router = APIRouter()
security = HTTPBearer()

SECRET = os.getenv("JWT_SECRET", "changeme")

class LoginInput(BaseModel):
    email: str
    password: str

# ✅ This is the correct login endpoint
@router.post("/login")
def login(data: LoginInput):
    print(f"🔐 Login attempt for {data.email}")
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s", (data.email,))
    user = cursor.fetchone()

    print(f"🧠 DB User Fetched: {user}")

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    raw_input = data.password.encode()
    stored_hash = user["password_hash"].encode()

    try:
        matched = bcrypt.checkpw(raw_input, stored_hash)
    except Exception as e:
        print(f"💥 bcrypt error: {e}")
        raise HTTPException(status_code=500, detail="bcrypt failure")

    if not matched:
        raise HTTPException(status_code=401, detail="Invalid password")

    print("✅ Password matched")

    stores: List[Dict[str, Any]] = []
    role = user.get("user_role") or "owner"

    try:
        cursor.execute(
            """
            SELECT
                us.id AS store_id,
                us.store_db,
                us.db_user,
                us.db_pass,
                us.store_name
            FROM user_stores us
            WHERE us.user_id = %s
            ORDER BY us.id
            """,
            (user["id"],),
        )
        for row in cursor.fetchall():
            stores.append(
                {
                    "store_id": row.get("store_id"),
                    "store_db": row.get("store_db"),
                    "db_user": row.get("db_user"),
                    "db_pass": row.get("db_pass"),
                    "store_name": row.get("store_name"),
                }
            )
    except Exception as e:
        print(f"⚠️ Unable to load store mappings from user_stores: {e}")

    if not stores:
        # Fallback to legacy single-store columns if they exist
        legacy_store_db = user.get("store_db")
        if legacy_store_db:
            stores.append(
                {
                    "store_id": None,
                    "store_db": legacy_store_db,
                    "db_user": user.get("db_user"),
                    "db_pass": user.get("db_pass"),
                    "store_name": user.get("store_name"),
                }
            )

    conn.close()

    if not stores and role != "admin":
        raise HTTPException(status_code=403, detail="No stores assigned to this user")

    primary_store = stores[0] if stores else {}

    token_payload = {
        "email": user["email"],
        "stores": stores,
        "role": role,
    }

    full_name = user.get("full_name")
    if full_name:
        token_payload["full_name"] = full_name

    if primary_store:
        token_payload.update(
            {
                "store_db": primary_store.get("store_db"),
                "db_user": primary_store.get("db_user"),
                "db_pass": primary_store.get("db_pass"),
            }
        )

    token = jwt.encode(token_payload, SECRET, algorithm="HS256")

    return {
        "token": token,
        "stores": stores,
        "role": role,
        "email": user["email"],
        "full_name": full_name,
    }

# ✅ This is the token auth dependency
def get_auth_user(token: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(token.credentials, SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
