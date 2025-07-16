from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import jwt, os, bcrypt
from app.db_core import get_core_connection

router = APIRouter()

SECRET = os.getenv("JWT_SECRET", "changeme")

class LoginInput(BaseModel):
    email: str
    password: str

@router.post("/login")
def login(data: LoginInput):
    print(f"🔐 Login attempt for {data.email}")
    conn = get_core_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s", (data.email,))
    user = cursor.fetchone()
    conn.close()

    print(f"🧠 DB User Fetched: {user}")

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    try:
        raw_input = data.password.encode()
        stored_hash = user["password_hash"].encode()

        print(f"🔍 Raw input: {raw_input}")
        print(f"🔐 Stored hash: {stored_hash}")

        if not bcrypt.checkpw(raw_input, stored_hash):
            print("❌ bcrypt.checkpw failed")
            raise HTTPException(status_code=401, detail="Invalid password")

        print("✅ Password matched")

    except Exception as e:
        print(f"💥 Exception during bcrypt check: {e}")
        raise HTTPException(status_code=500, detail="Internal bcrypt error")

    # Issue JWT
    token = jwt.encode({
        "email": user["email"],
        "store_db": user["store_db"],
        "db_user": user["db_user"],
        "db_pass": user["db_pass"]
    }, SECRET, algorithm="HS256")

    return {"token": token}

