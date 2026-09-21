from fastapi import APIRouter

from app.api.routes import ask, documents, health, search

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(ask.router)
api_router.include_router(documents.router)
api_router.include_router(search.router)
