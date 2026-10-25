import secrets
from typing import Annotated

from fastapi import Depends, Header

from app.api.deps import SettingsDep
from app.core.errors import AppError

API_KEY_HEADER = "X-API-Key"


def require_admin_key(
    settings: SettingsDep,
    api_key: Annotated[str | None, Header(alias=API_KEY_HEADER)] = None,
) -> None:
    expected = settings.admin_api_key.get_secret_value()
    if api_key is None or not secrets.compare_digest(api_key.encode(), expected.encode()):
        raise AppError(
            "unauthorized", f"Нужен корректный заголовок {API_KEY_HEADER}", status_code=401
        )


AdminKey = Depends(require_admin_key)
