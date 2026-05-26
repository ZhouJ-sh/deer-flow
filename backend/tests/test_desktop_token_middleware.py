import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.gateway.desktop_token_middleware import DesktopTokenMiddleware


def _app(token_file: Path | None = None) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        DesktopTokenMiddleware,
        enabled=True,
        token_file=str(token_file) if token_file else "",
    )

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/api/models")
    async def models():
        return {"models": []}

    @app.post("/api/v1/auth/initialize")
    async def initialize():
        return {"ok": True}

    @app.post("/api/v1/auth/login/local")
    async def login_local():
        return {"ok": True}

    @app.post("/api/v1/auth/register")
    async def register():
        return {"ok": True}

    @app.post("/api/v1/auth/logout")
    async def logout():
        return {"ok": True}

    return app


def test_health_is_exempt_when_token_file_is_valid(tmp_path: Path):
    token_file = tmp_path / "desktop-token"
    token_file.write_text("secret-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    client = TestClient(_app(token_file))

    assert client.get("/health").status_code == 200


def test_api_request_without_token_is_rejected(tmp_path: Path):
    token_file = tmp_path / "desktop-token"
    token_file.write_text("secret-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    client = TestClient(_app(token_file))

    response = client.get("/api/models")

    assert response.status_code == 403
    assert response.json()["detail"] == "Desktop token required"


def test_api_request_with_wrong_token_is_rejected(tmp_path: Path):
    token_file = tmp_path / "desktop-token"
    token_file.write_text("secret-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    client = TestClient(_app(token_file))

    response = client.get("/api/models", headers={"X-DeerFlow-Desktop-Token": "wrong"})

    assert response.status_code == 403


def test_api_request_with_non_ascii_token_is_rejected(tmp_path: Path):
    token_file = tmp_path / "desktop-token"
    token_file.write_text("secret-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    client = TestClient(_app(token_file))

    response = client.get("/api/models", headers=[(b"X-DeerFlow-Desktop-Token", "é".encode())])

    assert response.status_code == 403


def test_api_request_with_matching_token_is_allowed(tmp_path: Path):
    token_file = tmp_path / "desktop-token"
    token_file.write_text("secret-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    client = TestClient(_app(token_file))

    response = client.get("/api/models", headers={"X-DeerFlow-Desktop-Token": "secret-token"})

    assert response.status_code == 200


def test_auth_initialize_requires_desktop_token(tmp_path: Path):
    token_file = tmp_path / "desktop-token"
    token_file.write_text("secret-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    client = TestClient(_app(token_file))

    response = client.post("/api/v1/auth/initialize")

    assert response.status_code == 403


def test_setup_status_requires_desktop_token(tmp_path: Path):
    token_file = tmp_path / "desktop-token"
    token_file.write_text("secret-token\n", encoding="utf-8")
    token_file.chmod(0o600)

    app = _app(token_file)

    @app.get("/api/v1/auth/setup-status")
    async def setup_status():
        return {"needs_setup": True}

    client = TestClient(app)
    response = client.get("/api/v1/auth/setup-status")

    assert response.status_code == 403


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/api/v1/auth/login/local"),
        ("post", "/api/v1/auth/register"),
        ("post", "/api/v1/auth/logout"),
    ],
)
def test_public_auth_mutations_still_require_desktop_token(tmp_path: Path, method: str, path: str):
    token_file = tmp_path / "desktop-token"
    token_file.write_text("secret-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    client = TestClient(_app(token_file))

    response = getattr(client, method)(path)

    assert response.status_code == 403


def test_options_preflight_is_exempt(tmp_path: Path):
    token_file = tmp_path / "desktop-token"
    token_file.write_text("secret-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    client = TestClient(_app(token_file))

    assert client.options("/api/models").status_code in {200, 405}


def test_missing_token_file_fails_closed(tmp_path: Path):
    client = TestClient(_app(tmp_path / "missing-token"))

    with pytest.raises(RuntimeError, match="Desktop token file"):
        client.get("/health")


def test_empty_token_file_fails_closed(tmp_path: Path):
    token_file = tmp_path / "desktop-token"
    token_file.write_text("\n", encoding="utf-8")
    token_file.chmod(0o600)
    client = TestClient(_app(token_file))

    with pytest.raises(RuntimeError, match="Desktop token file is empty"):
        client.get("/health")


def test_world_readable_token_file_fails_closed_on_posix(tmp_path: Path):
    if os.name == "nt":
        return
    token_file = tmp_path / "desktop-token"
    token_file.write_text("secret-token\n", encoding="utf-8")
    token_file.chmod(0o644)
    client = TestClient(_app(token_file))

    with pytest.raises(RuntimeError, match="permissions"):
        client.get("/health")


def test_create_app_has_no_desktop_guard_by_default(monkeypatch: pytest.MonkeyPatch):
    from app.gateway.app import create_app

    monkeypatch.delenv("DEER_FLOW_DESKTOP", raising=False)
    monkeypatch.delenv("DEER_FLOW_DESKTOP_TOKEN_FILE", raising=False)
    client = TestClient(create_app())

    assert client.get("/health").status_code == 200


def test_create_app_desktop_mode_missing_token_file_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from app.gateway.app import create_app

    monkeypatch.setenv("DEER_FLOW_DESKTOP", "1")
    monkeypatch.setenv("DEER_FLOW_DESKTOP_TOKEN_FILE", str(tmp_path / "missing-token"))
    client = TestClient(create_app())

    with pytest.raises(RuntimeError, match="Desktop token file"):
        client.get("/health")


def test_create_app_desktop_mode_requires_token_before_public_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from app.gateway.app import create_app

    token_file = tmp_path / "desktop-token"
    token_file.write_text("secret-token\n", encoding="utf-8")
    token_file.chmod(0o600)
    monkeypatch.setenv("DEER_FLOW_DESKTOP", "1")
    monkeypatch.setenv("DEER_FLOW_DESKTOP_TOKEN_FILE", str(token_file))
    client = TestClient(create_app())

    assert client.get("/health").status_code == 200
    response = client.post("/api/v1/auth/initialize")

    assert response.status_code == 403
    assert response.json()["detail"] == "Desktop token required"
