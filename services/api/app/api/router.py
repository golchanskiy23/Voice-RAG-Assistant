from fastapi import APIRouter

from app.api.routes import ask, documents, health, search, telegram, voice

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(ask.router)
api_router.include_router(documents.router)
api_router.include_router(search.router)
api_router.include_router(voice.router)
api_router.include_router(telegram.router)
