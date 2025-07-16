from fastapi import FastAPI
from app.auth import router as auth_router
from app.insights import router as insights_router  
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path

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

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
