from fastapi import FastAPI
from app.auth import router as auth_router
from app.insights import router as insights_router
from app.admin import router as admin_router
from app.uploader import router as uploader_router
from app.sync import router as sync_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path

from app.bootstrap import bootstrap_admin_user

app = FastAPI(title="Store Insights API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000/dashboard"],  # or specify your frontend URL like ["https://yoursite.com"]
    allow_credentials=True,
    allow_methods=["*"],   # Includes OPTIONS
    allow_headers=["*"],
)



@app.get("/")
def root():
    return {"message": "Store Insights API is running"}

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(insights_router, prefix="/insights", tags=["insights"])
app.include_router(admin_router, prefix="/admin", tags=["admin"])
app.include_router(uploader_router, prefix="/uploader", tags=["uploader"])
app.include_router(sync_router, prefix="/sync", tags=["sync"])

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")


@app.on_event("startup")
def _bootstrap_admin_account() -> None:
    """Create or update the default admin account when credentials are provided."""

    bootstrap_admin_user()
