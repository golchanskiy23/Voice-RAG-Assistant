from fastapi import APIRouter

from app.api.routes import ask, documents, health, search, voice

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(ask.router)
api_router.include_router(documents.router)
api_router.include_router(search.router)
api_router.include_router(voice.router)
