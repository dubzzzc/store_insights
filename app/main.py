from fastapi import FastAPI, Request
from app.auth import router as auth_router
from app.insights import router as insights_router
from app.admin import router as admin_router
from app.uploader import router as uploader_router
from app.sync import router as sync_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import time

from app.bootstrap import bootstrap_admin_user

app = FastAPI(title="Store Insights API")


# Middleware to add no-cache headers for HTML files
class NoCacheHTMLMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path

        # Add no-cache headers for HTML files
        if (
            path.endswith(".html")
            or path == "/"
            or (path == "" and "text/html" in response.headers.get("content-type", ""))
        ):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            # Add ETag with timestamp to prevent caching
            response.headers["ETag"] = f'"{int(time.time())}"'
            # Prevent if-modified-since caching
            response.headers["Last-Modified"] = response.headers.get("Date", "")

        return response


app.add_middleware(NoCacheHTMLMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000/dashboard"],  # or specify your frontend URL like ["https://yoursite.com"]
    allow_credentials=True,
    allow_methods=["*"],   # Includes OPTIONS
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Spirit Store Insights API is running"}

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
