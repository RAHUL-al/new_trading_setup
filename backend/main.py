"""FastAPI main application — Trading Platform Backend."""

from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")  # Load from project root

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import os

from database import init_db
from routes import auth_routes, trading_routes, ws_routes, simulator_routes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

app = FastAPI(
    title="Trading Platform API",
    description="Multi-user NIFTY options trading platform with UT Bot signals",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_routes.router)
app.include_router(trading_routes.router)
app.include_router(ws_routes.router)
app.include_router(simulator_routes.router)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Trading Platform API is running"}


@app.get("/api/model-metadata")
def get_model_metadata():
    import json
    metadata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {"error": "metadata not found"}


# Serve dashboard at /dashboard
DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dashboard")
if os.path.isdir(DASHBOARD_DIR):
    @app.get("/dashboard")
    def serve_dashboard():
        return FileResponse(os.path.join(DASHBOARD_DIR, "index.html"))

    app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR), name="dashboard")
