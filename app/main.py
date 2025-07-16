from fastapi import FastAPI
from app.auth import router as auth_router
from app.insights import router as insights_router
from app.routers import insights

app = FastAPI(title="Store Insights API")

@app.get("/")
def root():
    return {"message": "Store Insights API is running"}

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(insights_router, prefix="/insights", tags=["insights"])
