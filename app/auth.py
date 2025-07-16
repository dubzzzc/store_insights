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
    print(f"🧠 DB User Fetched: {user}")
    conn.close()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    if not bcrypt.checkpw(data.password.encode(), user["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="Invalid password")

        token = jwt.encode({
        "email": user["email"],
        "store_db": user["store_db"],
        "db_user": user["db_user"],
        "db_pass": user["db_pass"]
    }, SECRET, algorithm="HS256")

    return {"token": token}
