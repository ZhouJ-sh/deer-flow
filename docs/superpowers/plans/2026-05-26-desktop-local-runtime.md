# Desktop Local Runtime Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Windows/macOS Electron shell that runs the existing DeerFlow Gateway, Agent loop, and local sandbox on the customer's machine while keeping upstream-tracked core changes minimal.

**Architecture:** Build a thin Electron app under `desktop/electron/` that owns desktop lifecycle, app-data paths, loopback ports, child sidecars, and a browser-facing HTTP proxy. Keep `RunManager`, `run_agent()`, `StreamBridge`, and `LocalSandboxProvider` behavior inside the existing Gateway/harness runtime. Add only one small, feature-flagged Gateway desktop-token guard outside `desktop/electron/`, with focused regression tests and a documented upstream touch list.

**Tech Stack:** Electron, TypeScript, Node.js HTTP proxy, Next standalone, FastAPI/Uvicorn, pytest, Vitest, Playwright/Electron smoke tests, electron-builder.

---

## Source Documents

- Design spec: `docs/superpowers/specs/2026-05-26-desktop-local-runtime-design.md`
- Plan: `docs/superpowers/plans/2026-05-26-desktop-local-runtime.md`

## Implementation Rules

- Keep all new desktop orchestration code under `desktop/electron/`.
- Do not rewrite or split `backend/packages/harness/deerflow/runtime/*`, `backend/packages/harness/deerflow/agents/*`, or `backend/packages/harness/deerflow/sandbox/*`.
- Every change outside `desktop/electron/` must be feature-flagged, tested, and recorded in the upstream touch list below.
- Use TDD for the Gateway hook and desktop library modules. For Electron smoke flows, add tests before wiring production launch paths where practical.
- Commit after each task using the commit message shown in that task.

## Upstream Touch List

| File | Touch Type | Reason | Removal Criteria |
|---|---|---|---|
| `backend/app/gateway/desktop_token_middleware.py` | Gateway hook, create | Reject direct loopback Gateway API calls in packaged desktop mode unless the desktop proxy injects `X-DeerFlow-Desktop-Token`. | Remove if Gateway gets an upstream-supported local transport guard or desktop mode is dropped. |
| `backend/app/gateway/app.py` | Gateway hook, small modify | Install the desktop token middleware only when `DEER_FLOW_DESKTOP=1`. | Remove with the middleware above. |
| `backend/tests/test_desktop_token_middleware.py` | Regression test, create | Proves token accepted/rejected behavior and exemption rules. | Remove with the middleware above. |
| `docs/desktop-local-runtime.md` | Additive docs, create | Documents desktop dev, packaging, data paths, token guard, and sandbox stance. | Keep while desktop shell exists. |
| `README.md` | Additive docs, small modify | Link to desktop runtime docs only. | Remove if desktop docs move elsewhere. |

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `backend/app/gateway/desktop_token_middleware.py` | Create | Feature-flagged local desktop token guard for Gateway requests. |
| `backend/app/gateway/app.py` | Modify | Add desktop token guard to FastAPI middleware stack only in desktop mode. |
| `backend/tests/test_desktop_token_middleware.py` | Create | Focused Gateway middleware tests. |
| `desktop/electron/package.json` | Create | Electron workspace scripts, dependencies, build config entrypoints. |
| `desktop/electron/.gitignore` | Create | Keep generated desktop artifacts out of upstream diffs. |
| `desktop/electron/tsup.config.ts` | Create | Build Electron main/preload/proxy/scripts into predictable `dist/` paths. |
| `desktop/electron/tsconfig.json` | Create | TypeScript compiler config for Electron main/proxy/tests. |
| `desktop/electron/vitest.config.ts` | Create | Unit-test config for desktop TypeScript modules. |
| `desktop/electron/electron-builder.yml` | Create | macOS/Windows packaging scaffold. |
| `desktop/electron/src/main/app-data.ts` | Create | Resolve OS app-data directory, create config/env/token/log directories. |
| `desktop/electron/src/main/env.ts` | Create | Parse app-data `.env`, generate persisted secrets, and build child env maps. |
| `desktop/electron/src/main/ports.ts` | Create | Allocate loopback ports and expose retry helpers. |
| `desktop/electron/src/main/paths.ts` | Create | Resolve dev and packaged resource paths without assuming CWD. |
| `desktop/electron/src/main/sidecar.ts` | Create | Child process abstraction with logs, readiness checks, cleanup. |
| `desktop/electron/src/main/runtime.ts` | Create | Start Gateway, Next standalone/dev frontend, and desktop proxy in order. |
| `desktop/electron/src/main/window.ts` | Create | BrowserWindow security settings and external-link handling. |
| `desktop/electron/src/main/index.ts` | Create | Electron app lifecycle entrypoint. |
| `desktop/electron/src/preload/index.ts` | Create | Tiny, token-free preload API for desktop status. |
| `desktop/electron/src/proxy/desktop-proxy.ts` | Create | Browser-facing proxy that injects the desktop token into Gateway traffic. |
| `desktop/electron/src/next/register-fetch.ts` | Create | Next server preload that marks SSR calls to `/_desktop-gateway` as internal. |
| `desktop/electron/src/scripts/build-runtime.ts` | Create | Stage backend/frontend/runtime artifacts into packaged resources. |
| `desktop/electron/src/scripts/stage-python-deps.ts` | Create | Copy a prebuilt dependency directory into `resources/backend/site-packages`. |
| `desktop/electron/src/scripts/smoke-packaged.ts` | Create | Cross-platform installed-path smoke helper. |
| `desktop/electron/tests/*.test.ts` | Create | Unit tests for app-data, env, ports, paths, sidecars, and proxy. |
| `desktop/electron/tests/electron-smoke.spec.ts` | Create | Electron launch smoke test. |
| `docs/desktop-local-runtime.md` | Create | Operator/dev documentation. |
| `README.md` | Modify | Add one link to the desktop docs. |

---

### Task 1: Gateway Desktop Token Guard

Add the only required backend hook. It is off by default and active only when `DEER_FLOW_DESKTOP=1`.

**Files:**
- Create: `backend/app/gateway/desktop_token_middleware.py`
- Modify: `backend/app/gateway/app.py`
- Test: `backend/tests/test_desktop_token_middleware.py`

- [ ] **Step 1: Write the failing middleware tests**

Create `backend/tests/test_desktop_token_middleware.py`:

```python
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && PYTHONPATH=. uv run pytest tests/test_desktop_token_middleware.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.gateway.desktop_token_middleware'`.

- [ ] **Step 3: Implement the middleware**

Create `backend/app/gateway/desktop_token_middleware.py`:

```python
from __future__ import annotations

import os
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

DESKTOP_TOKEN_HEADER = "X-DeerFlow-Desktop-Token"


class DesktopTokenMiddleware(BaseHTTPMiddleware):
    """Reject direct Gateway API calls in packaged desktop mode.

    This is a transport guard only. It does not replace DeerFlow user auth.
    """

    def __init__(self, app: ASGIApp, *, enabled: bool = False, token_file: str = "") -> None:
        super().__init__(app)
        self.enabled = enabled
        self.token_file = Path(token_file) if token_file else None
        self.expected_token = self._load_expected_token() if enabled else ""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if not self.enabled or self._is_exempt(request):
            return await call_next(request)

        supplied = request.headers.get(DESKTOP_TOKEN_HEADER, "")
        if not supplied or not secrets.compare_digest(self.expected_token, supplied):
            return JSONResponse(status_code=403, content={"detail": "Desktop token required"})

        return await call_next(request)

    @staticmethod
    def _is_exempt(request: Request) -> bool:
        path = request.url.path.rstrip("/") or "/"
        return request.method == "OPTIONS" or path == "/health"

    def _load_expected_token(self) -> str:
        if self.token_file is None:
            raise RuntimeError("Desktop token file is required when DEER_FLOW_DESKTOP=1")
        try:
            stat_result = self.token_file.stat()
        except OSError as exc:
            raise RuntimeError(f"Desktop token file is unreadable: {self.token_file}") from exc

        if os.name != "nt" and stat_result.st_mode & 0o077:
            raise RuntimeError("Desktop token file permissions must not allow group/other access")

        try:
            token = self.token_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(f"Desktop token file is unreadable: {self.token_file}") from exc
        if not token:
            raise RuntimeError("Desktop token file is empty")
        return token
```

- [ ] **Step 4: Wire the middleware in `create_app()`**

Modify `backend/app/gateway/app.py`:

```python
import os
```

Add the import:

```python
from app.gateway.desktop_token_middleware import DesktopTokenMiddleware
```

Add this immediately after `app.add_middleware(CSRFMiddleware)` and before the CORS block. This preserves the packaged request order `CORSMiddleware -> DesktopTokenMiddleware -> CSRFMiddleware -> AuthMiddleware` because Starlette wraps the most recently added middleware as the outer layer:

```python
    if os.getenv("DEER_FLOW_DESKTOP") == "1":
        app.add_middleware(
            DesktopTokenMiddleware,
            enabled=True,
            token_file=os.getenv("DEER_FLOW_DESKTOP_TOKEN_FILE", ""),
        )
```

- [ ] **Step 5: Run focused backend tests**

Run: `cd backend && PYTHONPATH=. uv run pytest tests/test_desktop_token_middleware.py -v`

Expected: PASS.

- [ ] **Step 6: Run auth/CSRF regression tests**

Run:

```bash
cd backend && PYTHONPATH=. uv run pytest \
  tests/test_desktop_token_middleware.py \
  tests/test_initialize_admin.py \
  -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/gateway/app.py backend/app/gateway/desktop_token_middleware.py backend/tests/test_desktop_token_middleware.py
git commit -m "feat: add desktop gateway token guard"
```

---

### Task 2: Electron Package Scaffold

Create the desktop workspace without touching root package management. Keep it self-contained in `desktop/electron/`.

**Files:**
- Create: `desktop/electron/package.json`
- Create: `desktop/electron/.gitignore`
- Create: `desktop/electron/tsup.config.ts`
- Create: `desktop/electron/tsconfig.json`
- Create: `desktop/electron/vitest.config.ts`
- Create: `desktop/electron/electron-builder.yml`
- Create: `desktop/electron/src/main/index.ts`
- Create: `desktop/electron/src/preload/index.ts`
- Test: `desktop/electron/tests/scaffold.test.ts`

- [ ] **Step 1: Create a minimal failing scaffold test**

Create `desktop/electron/tests/scaffold.test.ts`:

```ts
import { describe, expect, test } from "vitest";

describe("desktop scaffold", () => {
  test("sets the desktop app name", async () => {
    const pkg = await import("../package.json", { assert: { type: "json" } });
    expect(pkg.default.name).toBe("@deer-flow/desktop-electron");
  });
});
```

- [ ] **Step 2: Create `package.json`**

Create `desktop/electron/package.json`:

```json
{
  "name": "@deer-flow/desktop-electron",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "main": "dist/main/index.js",
  "scripts": {
    "build": "tsc --noEmit -p tsconfig.json && pnpm build:bundle",
    "build:bundle": "tsup --config tsup.config.ts",
    "dev": "pnpm build && electron .",
    "pack": "electron-builder --dir",
    "dist": "electron-builder",
    "test": "vitest run",
    "test:watch": "vitest",
    "smoke": "node dist/scripts/smoke-packaged.js"
  },
  "dependencies": {
    "@electron-toolkit/preload": "^3.0.2",
    "@electron-toolkit/utils": "^4.0.0",
    "dotenv": "^17.2.3",
    "get-port": "^7.1.0",
    "http-proxy": "^1.18.1",
    "mime": "^4.1.0",
    "yaml": "^2.8.3"
  },
  "devDependencies": {
    "@types/http-proxy": "^1.17.16",
    "@types/mime": "^3.0.4",
    "@types/node": "^20.14.10",
    "electron": "^39.2.7",
    "electron-builder": "^26.0.12",
    "tsup": "^8.5.1",
    "typescript": "^5.8.2",
    "vitest": "^4.1.4"
  }
}
```

- [ ] **Step 3: Create ignore, build, TypeScript, and Vitest config**

Create `desktop/electron/.gitignore`:

```gitignore
node_modules/
dist/
resources/
release/
*.log
```

Create `desktop/electron/tsup.config.ts`:

```ts
import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    "main/index": "src/main/index.ts",
    "preload/index": "src/preload/index.ts",
  },
  format: ["esm"],
  platform: "node",
  target: "node20",
  outDir: "dist",
  clean: true,
  splitting: false,
  sourcemap: true,
  external: ["electron"],
});
```

Create `desktop/electron/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "esModuleInterop": true,
    "resolveJsonModule": true,
    "skipLibCheck": true,
    "outDir": "dist",
    "rootDir": ".",
    "types": ["node", "vitest/globals"],
    "allowSyntheticDefaultImports": true,
    "verbatimModuleSyntax": false
  },
  "include": ["src/**/*.ts", "tests/**/*.ts", "tsup.config.ts", "vitest.config.ts"]
}
```

Create `desktop/electron/vitest.config.ts`:

```ts
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
});
```

- [ ] **Step 4: Create builder scaffold**

Create `desktop/electron/electron-builder.yml`:

```yaml
appId: sh.zhouj.deerflow.desktop
productName: DeerFlow
directories:
  output: release
files:
  - dist/**
  - package.json
extraResources:
  - from: resources
    to: .
    filter:
      - "**/*"
mac:
  target:
    - dmg
    - zip
win:
  target:
    - nsis
    - zip
nsis:
  oneClick: false
  perMachine: false
  allowToChangeInstallationDirectory: true
```

- [ ] **Step 5: Create initial Electron entrypoints**

Create `desktop/electron/src/main/index.ts`:

```ts
import { app, BrowserWindow } from "electron";

async function main() {
  await app.whenReady();
  const win = new BrowserWindow({
    width: 1280,
    height: 840,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  await win.loadURL("about:blank");
}

void main();
```

Create `desktop/electron/src/preload/index.ts`:

```ts
import { contextBridge } from "electron";

contextBridge.exposeInMainWorld("deerFlowDesktop", {
  version: "0.1.0",
});
```

- [ ] **Step 6: Install desktop dependencies**

Run: `cd desktop/electron && pnpm install`

Expected: `pnpm-lock.yaml` is created under `desktop/electron/` and dependencies install successfully.

- [ ] **Step 7: Run scaffold tests and build**

Run:

```bash
cd desktop/electron
pnpm test
pnpm build
```

Expected: PASS and `dist/main/index.js` / `dist/preload/index.js` exist.

- [ ] **Step 8: Commit**

```bash
git add desktop/electron
git commit -m "feat: scaffold electron desktop shell"
```

---

### Task 3: Desktop App-Data, Token, Config, and Env Manager

Implement the desktop data directory and generated config files. This task is library-only and does not launch sidecars.

**Files:**
- Create: `desktop/electron/src/main/app-data.ts`
- Create: `desktop/electron/src/main/env.ts`
- Create: `desktop/electron/tests/app-data.test.ts`
- Create: `desktop/electron/tests/env.test.ts`

- [ ] **Step 1: Write failing app-data tests**

Create `desktop/electron/tests/app-data.test.ts`:

```ts
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, test } from "vitest";

import { ensureDesktopData, type DesktopDataPaths } from "../src/main/app-data";

let created: string[] = [];

async function tempRoot() {
  const dir = await mkdtemp(join(tmpdir(), "deerflow-desktop-"));
  created.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(created.map((dir) => rm(dir, { recursive: true, force: true })));
  created = [];
});

describe("ensureDesktopData", () => {
  test("creates stable desktop data paths", async () => {
    const root = await tempRoot();
    const paths: DesktopDataPaths = await ensureDesktopData({ root });

    expect(paths.root).toBe(root);
    await expect(stat(paths.deerFlowHome)).resolves.toBeDefined();
    await expect(stat(paths.logsDir)).resolves.toBeDefined();
    await expect(stat(paths.runtimeDir)).resolves.toBeDefined();
    await expect(stat(paths.configPath)).resolves.toBeDefined();
    await expect(stat(paths.extensionsConfigPath)).resolves.toBeDefined();
    await expect(stat(paths.envPath)).resolves.toBeDefined();
    await expect(stat(paths.tokenPath)).resolves.toBeDefined();
  });

  test("desktop config uses absolute sqlite_dir and local sandbox default", async () => {
    const root = await tempRoot();
    const exampleDir = await tempRoot();
    await writeFile(
      join(exampleDir, "config.example.yaml"),
      [
        "config_version: 10",
        "log_level: info",
        "models: []",
        "tools:",
        "  - name: web_search",
        "    group: web",
        "database:",
        "  backend: postgres",
        "run_events:",
        "  backend: stream",
        "sandbox:",
        "  use: deerflow.community.aio_sandbox:AioSandboxProvider",
        "  container_prefix: deer-flow-sandbox",
        "",
      ].join("\n"),
      "utf-8",
    );
    await writeFile(
      join(exampleDir, "extensions_config.example.json"),
      '{\n  "mcpServers": {"github": {"enabled": false}},\n  "skills": {}\n}\n',
      "utf-8",
    );

    const paths = await ensureDesktopData({
      root,
      exampleConfigPath: join(exampleDir, "config.example.yaml"),
      exampleExtensionsConfigPath: join(exampleDir, "extensions_config.example.json"),
    });
    const config = await readFile(paths.configPath, "utf-8");
    const extensionsConfig = await readFile(paths.extensionsConfigPath, "utf-8");

    expect(config).toContain("name: web_search");
    expect(config).toContain("backend: sqlite");
    expect(config).toContain(`sqlite_dir: ${paths.sqliteDir}`);
    expect(config).toContain("backend: db");
    expect(config).toContain("use: deerflow.sandbox.local:LocalSandboxProvider");
    expect(config).toContain("allow_host_bash: false");
    expect(config).not.toContain("container_prefix");
    expect(extensionsConfig).toContain('"github"');
  });

  test("existing config is re-sanitized to desktop-only local sandbox", async () => {
    const root = await tempRoot();
    await writeFile(
      join(root, "config.yaml"),
      [
        "config_version: 10",
        "log_level: debug",
        "database:",
        "  backend: postgres",
        "checkpointer:",
        "  backend: postgres",
        "run_events:",
        "  backend: stream",
        "sandbox:",
        "  use: deerflow.community.aio_sandbox:AioSandboxProvider",
        "  container_prefix: stale-desktop-container",
        "",
      ].join("\n"),
      "utf-8",
    );

    const paths = await ensureDesktopData({ root });
    const config = await readFile(paths.configPath, "utf-8");

    expect(config).toContain("log_level: debug");
    expect(config).toContain("backend: sqlite");
    expect(config).toContain(`sqlite_dir: ${paths.sqliteDir}`);
    expect(config).toContain("backend: db");
    expect(config).toContain("use: deerflow.sandbox.local:LocalSandboxProvider");
    expect(config).toContain("allow_host_bash: false");
    expect(config).not.toContain("AioSandboxProvider");
    expect(config).not.toContain("container_prefix");
    expect(config).not.toContain("checkpointer");
  });
});
```

- [ ] **Step 2: Implement app-data manager**

Create `desktop/electron/src/main/app-data.ts`:

```ts
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { randomBytes, randomUUID } from "node:crypto";
import YAML from "yaml";

export interface EnsureDesktopDataOptions {
  root: string;
  exampleConfigPath?: string;
  exampleExtensionsConfigPath?: string;
}

export interface DesktopDataPaths {
  root: string;
  deerFlowHome: string;
  dataDir: string;
  logsDir: string;
  runtimeDir: string;
  sqliteDir: string;
  exampleConfigPath: string | null;
  exampleExtensionsConfigPath: string | null;
  configPath: string;
  extensionsConfigPath: string;
  envPath: string;
  tokenPath: string;
  installIdPath: string;
}

export async function ensureDesktopData(options: EnsureDesktopDataOptions): Promise<DesktopDataPaths> {
  const root = options.root;
  const paths: DesktopDataPaths = {
    root,
    deerFlowHome: join(root, ".deer-flow"),
    dataDir: join(root, ".deer-flow", "data"),
    logsDir: join(root, "logs"),
    runtimeDir: join(root, "runtime"),
    sqliteDir: join(root, ".deer-flow", "data"),
    exampleConfigPath: options.exampleConfigPath ?? null,
    exampleExtensionsConfigPath: options.exampleExtensionsConfigPath ?? null,
    configPath: join(root, "config.yaml"),
    extensionsConfigPath: join(root, "extensions_config.json"),
    envPath: join(root, ".env"),
    tokenPath: join(root, "desktop-token"),
    installIdPath: join(root, "install-id"),
  };

  await mkdir(paths.dataDir, { recursive: true });
  await mkdir(paths.logsDir, { recursive: true });
  await mkdir(paths.runtimeDir, { recursive: true });

  const installId = await ensureTextFile(paths.installIdPath, `${randomUUID()}\n`);
  await ensureTextFile(paths.tokenPath, `${randomBytes(32).toString("base64url")}\n`);
  await ensureTextFile(paths.envPath, "");
  await ensureTextFile(
    paths.extensionsConfigPath,
    await readExampleOrDefault(options.exampleExtensionsConfigPath, '{\n  "mcpServers": {},\n  "skills": {}\n}\n'),
  );
  await writeDesktopConfig(paths, installId.trim(), options.exampleConfigPath);

  return paths;
}

async function ensureTextFile(path: string, defaultContent: string): Promise<string> {
  try {
    return await readFile(path, "utf-8");
  } catch {
    await writeFile(path, defaultContent, { encoding: "utf-8", mode: 0o600 });
    return defaultContent;
  }
}

async function readExampleOrDefault(path: string | undefined, fallback: string): Promise<string> {
  if (!path) return fallback;
  try {
    return await readFile(path, "utf-8");
  } catch {
    return fallback;
  }
}

async function writeDesktopConfig(
  paths: DesktopDataPaths,
  installId: string,
  exampleConfigPath?: string,
): Promise<void> {
  const existingOrExample = await readExistingConfigOrExample(paths.configPath, exampleConfigPath);
  await writeFile(paths.configPath, buildDesktopConfig(paths, installId, existingOrExample), {
    encoding: "utf-8",
    mode: 0o600,
  });
}

async function readExistingConfigOrExample(configPath: string, exampleConfigPath?: string): Promise<string> {
  try {
    return await readFile(configPath, "utf-8");
  } catch {
    return readExampleOrDefault(
      exampleConfigPath,
      "config_version: 10\nlog_level: info\nmodels: []\ntools: []\n",
    );
  }
}

function buildDesktopConfig(
  paths: DesktopDataPaths,
  _installId: string,
  rawConfig: string,
): string {
  const config = (YAML.parse(rawConfig) ?? {}) as Record<string, unknown>;
  config.database = {
    ...((config.database as Record<string, unknown> | undefined) ?? {}),
    backend: "sqlite",
    sqlite_dir: paths.sqliteDir,
  };
  config.run_events = {
    ...((config.run_events as Record<string, unknown> | undefined) ?? {}),
    backend: "db",
  };
  delete config.checkpointer;
  config.sandbox = {
    use: "deerflow.sandbox.local:LocalSandboxProvider",
    allow_host_bash: false,
  };
  return `${YAML.stringify(config)}\n`;
}
```

- [ ] **Step 3: Write failing env tests**

Create `desktop/electron/tests/env.test.ts`:

```ts
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, test } from "vitest";

import { ensureDesktopData } from "../src/main/app-data";
import { buildGatewayEnv, buildNextEnv, loadDesktopDotEnv } from "../src/main/env";

let created: string[] = [];

async function tempRoot() {
  const dir = await mkdtemp(join(tmpdir(), "deerflow-env-"));
  created.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(created.map((dir) => rm(dir, { recursive: true, force: true })));
  created = [];
});

describe("desktop env", () => {
  test("parses app-data .env without mutating process.env", async () => {
    const root = await tempRoot();
    const paths = await ensureDesktopData({ root });
    delete process.env.DEER_FLOW_DESKTOP_TEST_KEY;
    await writeFile(paths.envPath, "DEER_FLOW_DESKTOP_TEST_KEY=sk-test\nEMPTY=\n", "utf-8");

    const env = await loadDesktopDotEnv(paths.envPath);

    expect(env.DEER_FLOW_DESKTOP_TEST_KEY).toBe("sk-test");
    expect(process.env.DEER_FLOW_DESKTOP_TEST_KEY).toBeUndefined();
  });

  test("builds gateway and next envs with desktop paths", async () => {
    const root = await tempRoot();
    const paths = await ensureDesktopData({ root });

    const gatewayEnv = await buildGatewayEnv(paths, { frontendOrigin: "http://127.0.0.1:43000" });
    const nextEnv = await buildNextEnv(paths, { proxyOrigin: "http://127.0.0.1:43000" });

    expect(gatewayEnv.DEER_FLOW_DESKTOP).toBe("1");
    expect(gatewayEnv.DEER_FLOW_HOME).toBe(paths.deerFlowHome);
    expect(gatewayEnv.DEER_FLOW_CONFIG_PATH).toBe(paths.configPath);
    expect(gatewayEnv.DEER_FLOW_EXTENSIONS_CONFIG_PATH).toBe(paths.extensionsConfigPath);
    expect(gatewayEnv.DEER_FLOW_DESKTOP_TOKEN_FILE).toBe(paths.tokenPath);
    expect(gatewayEnv.GATEWAY_CORS_ORIGINS).toBe("http://127.0.0.1:43000");
    expect(nextEnv.DEER_FLOW_INTERNAL_GATEWAY_BASE_URL).toBe("http://127.0.0.1:43000/_desktop-gateway");
    expect(nextEnv.NEXT_PUBLIC_LANGGRAPH_BASE_URL).toBeUndefined();
  });
});
```

- [ ] **Step 4: Implement env manager**

Create `desktop/electron/src/main/env.ts`:

```ts
import { readFile, writeFile } from "node:fs/promises";
import { randomBytes } from "node:crypto";
import { join } from "node:path";
import { parse } from "dotenv";

import type { DesktopDataPaths } from "./app-data";

export async function loadDesktopDotEnv(envPath: string): Promise<Record<string, string>> {
  try {
    return parse(await readFile(envPath, "utf-8"));
  } catch {
    return {};
  }
}

export async function buildGatewayEnv(
  paths: DesktopDataPaths,
  options: { frontendOrigin: string },
): Promise<NodeJS.ProcessEnv> {
  return {
    ...process.env,
    ...(await loadDesktopDotEnv(paths.envPath)),
    DEER_FLOW_DESKTOP: "1",
    DEER_FLOW_HOME: paths.deerFlowHome,
    DEER_FLOW_CONFIG_PATH: paths.configPath,
    DEER_FLOW_EXTENSIONS_CONFIG_PATH: paths.extensionsConfigPath,
    DEER_FLOW_PROJECT_ROOT: paths.root,
    DEER_FLOW_DESKTOP_TOKEN_FILE: paths.tokenPath,
    GATEWAY_HOST: "127.0.0.1",
    GATEWAY_CORS_ORIGINS: options.frontendOrigin,
  };
}

export async function buildNextEnv(
  paths: DesktopDataPaths,
  options: { proxyOrigin: string },
): Promise<NodeJS.ProcessEnv> {
  const betterAuthSecret = await ensureBetterAuthSecret(paths);
  const env = {
    ...process.env,
    ...(await loadDesktopDotEnv(paths.envPath)),
    BETTER_AUTH_SECRET: betterAuthSecret,
    DEER_FLOW_INTERNAL_GATEWAY_BASE_URL: `${options.proxyOrigin}/_desktop-gateway`,
    DEER_FLOW_TRUSTED_ORIGINS: options.proxyOrigin,
    SKIP_ENV_VALIDATION: "1",
  };
  delete env.NEXT_PUBLIC_LANGGRAPH_BASE_URL;
  delete env.NEXT_PUBLIC_BACKEND_BASE_URL;
  return env;
}

async function ensureBetterAuthSecret(paths: DesktopDataPaths): Promise<string> {
  const secretPath = join(paths.root, "better-auth-secret");
  try {
    const value = (await readFile(secretPath, "utf-8")).trim();
    if (value) return value;
  } catch {
    // create below
  }
  const value = randomBytes(32).toString("base64url");
  await writeFile(secretPath, `${value}\n`, { encoding: "utf-8", mode: 0o600 });
  return value;
}
```

- [ ] **Step 5: Run desktop unit tests**

Run: `cd desktop/electron && pnpm test`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add desktop/electron/src/main/app-data.ts desktop/electron/src/main/env.ts desktop/electron/tests/app-data.test.ts desktop/electron/tests/env.test.ts
git commit -m "feat: add desktop app data manager"
```

---

### Task 4: Ports, Paths, and Sidecar Process Manager

Build reusable launch primitives before wiring the runtime.

**Files:**
- Create: `desktop/electron/src/main/ports.ts`
- Create: `desktop/electron/src/main/paths.ts`
- Create: `desktop/electron/src/main/sidecar.ts`
- Create: `desktop/electron/tests/ports.test.ts`
- Create: `desktop/electron/tests/paths.test.ts`
- Create: `desktop/electron/tests/sidecar.test.ts`

- [ ] **Step 1: Write failing port and path tests**

Create `desktop/electron/tests/ports.test.ts`:

```ts
import { createServer } from "node:net";
import { describe, expect, test } from "vitest";

import { allocateDesktopPorts } from "../src/main/ports";

describe("allocateDesktopPorts", () => {
  test("returns three distinct loopback ports", async () => {
    const ports = await allocateDesktopPorts();
    expect(new Set([ports.gatewayPort, ports.nextPort, ports.proxyPort]).size).toBe(3);
  });

  test("allocated ports can be bound on 127.0.0.1", async () => {
    const ports = await allocateDesktopPorts();
    await new Promise<void>((resolve, reject) => {
      const server = createServer();
      server.once("error", reject);
      server.listen(ports.gatewayPort, "127.0.0.1", () => {
        server.close(() => resolve());
      });
    });
  });
});
```

Create `desktop/electron/tests/paths.test.ts`:

```ts
import { describe, expect, test } from "vitest";

import { resolveDesktopResources } from "../src/main/paths";

describe("resolveDesktopResources", () => {
  test("resolves dev resources from repo root", () => {
    const paths = resolveDesktopResources({
      packaged: false,
      appPath: "/repo/desktop/electron",
      resourcesPath: "/ignored",
    });

    expect(paths.repoRoot).toMatch(/[/\\]repo$/);
    expect(paths.backendDir).toMatch(/[/\\]repo[/\\]backend$/);
    expect(paths.frontendDir).toMatch(/[/\\]repo[/\\]frontend$/);
  });
});
```

- [ ] **Step 2: Implement ports and paths**

Create `desktop/electron/src/main/ports.ts`:

```ts
import getPort from "get-port";

export interface DesktopPorts {
  gatewayPort: number;
  nextPort: number;
  proxyPort: number;
}

export async function allocateDesktopPorts(): Promise<DesktopPorts> {
  const gatewayPort = await getPort({ host: "127.0.0.1" });
  const nextPort = await getPort({ host: "127.0.0.1", exclude: [gatewayPort] });
  const proxyPort = await getPort({ host: "127.0.0.1", exclude: [gatewayPort, nextPort] });
  return { gatewayPort, nextPort, proxyPort };
}
```

Create `desktop/electron/src/main/paths.ts`:

```ts
import { join, resolve } from "node:path";

export interface DesktopResourceOptions {
  packaged: boolean;
  appPath: string;
  resourcesPath: string;
}

export interface DesktopResources {
  repoRoot: string | null;
  backendDir: string;
  frontendDir: string;
  pythonBin: string;
  nodeBin: string;
  desktopServerDir: string;
}

export function resolveDesktopResources(options: DesktopResourceOptions): DesktopResources {
  if (!options.packaged) {
    const repoRoot = resolve(options.appPath, "..", "..");
    return {
      repoRoot,
      backendDir: join(repoRoot, "backend"),
      frontendDir: join(repoRoot, "frontend"),
      pythonBin: "python",
      nodeBin: "node",
      desktopServerDir: join(options.appPath, "dist"),
    };
  }

  const resources = options.resourcesPath;
  return {
    repoRoot: null,
    backendDir: join(resources, "backend"),
    frontendDir: join(resources, "frontend"),
    pythonBin: process.platform === "win32"
      ? join(resources, "runtimes", "python", "python.exe")
      : join(resources, "runtimes", "python", "bin", "python"),
    nodeBin: process.platform === "win32"
      ? join(resources, "runtimes", "node", "node.exe")
      : join(resources, "runtimes", "node", "bin", "node"),
    desktopServerDir: join(resources, "desktop-server"),
  };
}
```

- [ ] **Step 3: Write failing sidecar tests**

Create `desktop/electron/tests/sidecar.test.ts`:

```ts
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, test } from "vitest";

import { startSidecar } from "../src/main/sidecar";

let created: string[] = [];

afterEach(async () => {
  await Promise.all(created.map((dir) => rm(dir, { recursive: true, force: true })));
  created = [];
});

describe("startSidecar", () => {
  test("captures stdout and exits cleanly", async () => {
    const dir = await mkdtemp(join(tmpdir(), "deerflow-sidecar-"));
    created.push(dir);
    const sidecar = startSidecar({
      name: "node-hello",
      command: process.execPath,
      args: ["-e", "console.log('hello sidecar')"],
      cwd: dir,
      env: process.env,
      logPath: join(dir, "sidecar.log"),
    });

    const code = await sidecar.exit;
    expect(code).toBe(0);
    await expect(readFile(join(dir, "sidecar.log"), "utf-8")).resolves.toContain("hello sidecar");
  });
});
```

- [ ] **Step 4: Implement sidecar manager**

Create `desktop/electron/src/main/sidecar.ts`:

```ts
import { createWriteStream } from "node:fs";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";

export interface SidecarOptions {
  name: string;
  command: string;
  args: string[];
  cwd: string;
  env: NodeJS.ProcessEnv;
  logPath: string;
}

export interface SidecarProcess {
  name: string;
  child: ChildProcessWithoutNullStreams;
  exit: Promise<number | null>;
  stop: () => Promise<void>;
}

export function startSidecar(options: SidecarOptions): SidecarProcess {
  const log = createWriteStream(options.logPath, { flags: "a" });
  const child = spawn(options.command, options.args, {
    cwd: options.cwd,
    env: options.env,
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });

  child.stdout.pipe(log, { end: false });
  child.stderr.pipe(log, { end: false });

  const exit = new Promise<number | null>((resolve, reject) => {
    child.once("error", (err) => {
      log.end();
      reject(err);
    });
    child.once("exit", (code) => {
      log.end();
      resolve(code);
    });
  });

  async function stop() {
    if (child.exitCode !== null || child.signalCode !== null) return;
    child.kill(process.platform === "win32" ? undefined : "SIGTERM");
    await Promise.race([
      exit,
      new Promise<void>((resolve) => setTimeout(resolve, 5000)),
    ]);
    if (child.exitCode === null && child.signalCode === null) {
      child.kill("SIGKILL");
    }
  }

  return { name: options.name, child, exit, stop };
}
```

- [ ] **Step 5: Run desktop unit tests**

Run: `cd desktop/electron && pnpm test -- tests/ports.test.ts tests/paths.test.ts tests/sidecar.test.ts`

Expected: PASS.

- [ ] **Step 6: Run all desktop unit tests**

Run: `cd desktop/electron && pnpm test`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add desktop/electron/src/main/ports.ts desktop/electron/src/main/paths.ts desktop/electron/src/main/sidecar.ts desktop/electron/tests/ports.test.ts desktop/electron/tests/paths.test.ts desktop/electron/tests/sidecar.test.ts
git commit -m "feat: add desktop sidecar primitives"
```

---

### Task 5: Desktop HTTP Proxy With Token Injection

Implement the browser-facing proxy. It forwards frontend pages/static assets to Next and Gateway API calls to the local Gateway, injecting the desktop token server-side only.

**Files:**
- Create: `desktop/electron/src/proxy/desktop-proxy.ts`
- Modify: `desktop/electron/tsup.config.ts`
- Create: `desktop/electron/tests/proxy.test.ts`

- [ ] **Step 1: Write failing proxy tests**

Create `desktop/electron/tests/proxy.test.ts`:

```ts
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, test } from "vitest";

import { startDesktopProxy } from "../src/proxy/desktop-proxy";
import { allocateDesktopPorts } from "../src/main/ports";

let cleanups: Array<() => Promise<void> | void> = [];

afterEach(async () => {
  for (const cleanup of cleanups.reverse()) await cleanup();
  cleanups = [];
});

function startJsonServer(handler: (req: IncomingMessage, res: ServerResponse) => void): Promise<{ port: number; close: () => Promise<void> }> {
  return new Promise(async (resolve) => {
    const { gatewayPort: port } = await allocateDesktopPorts();
    const server = createServer(handler);
    server.listen(port, "127.0.0.1", () => {
      resolve({
        port,
        close: () => new Promise<void>((done) => server.close(() => done())),
      });
    });
  });
}

describe("desktop proxy", () => {
  test("injects desktop token for api requests", async () => {
    const dir = await mkdtemp(join(tmpdir(), "deerflow-proxy-"));
    cleanups.push(() => rm(dir, { recursive: true, force: true }));
    const tokenPath = join(dir, "desktop-token");
    await writeFile(tokenPath, "secret-token\n", "utf-8");

    const gateway = await startJsonServer((req, res) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ token: req.headers["x-deerflow-desktop-token"] }));
    });
    cleanups.push(gateway.close);

    const next = await startJsonServer((_req, res) => res.end("next"));
    cleanups.push(next.close);

    const { proxyPort } = await allocateDesktopPorts();
    const proxy = await startDesktopProxy({
      host: "127.0.0.1",
      port: proxyPort,
      gatewayOrigin: `http://127.0.0.1:${gateway.port}`,
      nextOrigin: `http://127.0.0.1:${next.port}`,
      tokenPath,
      internalHeaderValue: "internal-secret",
      logPath: join(dir, "desktop-proxy.log"),
    });
    cleanups.push(proxy.close);

    const res = await fetch(`http://127.0.0.1:${proxyPort}/api/models`);
    expect(await res.json()).toEqual({ token: "secret-token" });
  });

  test("rewrites langgraph api prefix before forwarding to Gateway", async () => {
    const dir = await mkdtemp(join(tmpdir(), "deerflow-proxy-"));
    cleanups.push(() => rm(dir, { recursive: true, force: true }));
    const tokenPath = join(dir, "desktop-token");
    await writeFile(tokenPath, "secret-token\n", "utf-8");
    const gateway = await startJsonServer((req, res) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ url: req.url }));
    });
    cleanups.push(gateway.close);
    const next = await startJsonServer((_req, res) => res.end("next"));
    cleanups.push(next.close);
    const { proxyPort } = await allocateDesktopPorts();
    const proxy = await startDesktopProxy({
      host: "127.0.0.1",
      port: proxyPort,
      gatewayOrigin: `http://127.0.0.1:${gateway.port}`,
      nextOrigin: `http://127.0.0.1:${next.port}`,
      tokenPath,
      internalHeaderValue: "internal-secret",
      logPath: join(dir, "desktop-proxy.log"),
    });
    cleanups.push(proxy.close);

    const res = await fetch(`http://127.0.0.1:${proxyPort}/api/langgraph/runs?stream=true`);

    expect(await res.json()).toEqual({ url: "/api/runs?stream=true" });
  });

  test("rewrites bare langgraph api prefix with query string", async () => {
    const dir = await mkdtemp(join(tmpdir(), "deerflow-proxy-"));
    cleanups.push(() => rm(dir, { recursive: true, force: true }));
    const tokenPath = join(dir, "desktop-token");
    await writeFile(tokenPath, "secret-token\n", "utf-8");
    const gateway = await startJsonServer((req, res) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ url: req.url }));
    });
    cleanups.push(gateway.close);
    const next = await startJsonServer((_req, res) => res.end("next"));
    cleanups.push(next.close);
    const { proxyPort } = await allocateDesktopPorts();
    const proxy = await startDesktopProxy({
      host: "127.0.0.1",
      port: proxyPort,
      gatewayOrigin: `http://127.0.0.1:${gateway.port}`,
      nextOrigin: `http://127.0.0.1:${next.port}`,
      tokenPath,
      internalHeaderValue: "internal-secret",
      logPath: join(dir, "desktop-proxy.log"),
    });
    cleanups.push(proxy.close);

    const res = await fetch(`http://127.0.0.1:${proxyPort}/api/langgraph?foo=bar`);

    expect(await res.json()).toEqual({ url: "/api?foo=bar" });
  });

  test("rejects browser access to _desktop-gateway prefix", async () => {
    const dir = await mkdtemp(join(tmpdir(), "deerflow-proxy-"));
    cleanups.push(() => rm(dir, { recursive: true, force: true }));
    const tokenPath = join(dir, "desktop-token");
    await writeFile(tokenPath, "secret-token\n", "utf-8");
    const gateway = await startJsonServer((_req, res) => res.end("gateway"));
    cleanups.push(gateway.close);
    const next = await startJsonServer((_req, res) => res.end("next"));
    cleanups.push(next.close);
    const { proxyPort } = await allocateDesktopPorts();
    const proxy = await startDesktopProxy({
      host: "127.0.0.1",
      port: proxyPort,
      gatewayOrigin: `http://127.0.0.1:${gateway.port}`,
      nextOrigin: `http://127.0.0.1:${next.port}`,
      tokenPath,
      internalHeaderValue: "internal-secret",
      logPath: join(dir, "desktop-proxy.log"),
    });
    cleanups.push(proxy.close);

    const res = await fetch(`http://127.0.0.1:${proxyPort}/_desktop-gateway/api/models`, {
      headers: {
        "sec-fetch-mode": "cors",
        "x-deerflow-desktop-internal-next": "1",
      },
    });

    expect(res.status).toBe(404);
  });

  test("allows internal next access to _desktop-gateway prefix", async () => {
    const dir = await mkdtemp(join(tmpdir(), "deerflow-proxy-"));
    cleanups.push(() => rm(dir, { recursive: true, force: true }));
    const tokenPath = join(dir, "desktop-token");
    await writeFile(tokenPath, "secret-token\n", "utf-8");
    const gateway = await startJsonServer((req, res) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ url: req.url, token: req.headers["x-deerflow-desktop-token"] }));
    });
    cleanups.push(gateway.close);
    const next = await startJsonServer((_req, res) => res.end("next"));
    cleanups.push(next.close);
    const { proxyPort } = await allocateDesktopPorts();
    const proxy = await startDesktopProxy({
      host: "127.0.0.1",
      port: proxyPort,
      gatewayOrigin: `http://127.0.0.1:${gateway.port}`,
      nextOrigin: `http://127.0.0.1:${next.port}`,
      tokenPath,
      internalHeaderValue: "internal-secret",
      logPath: join(dir, "desktop-proxy.log"),
    });
    cleanups.push(proxy.close);

    const res = await fetch(`http://127.0.0.1:${proxyPort}/_desktop-gateway/api/v1/auth/setup-status`, {
      headers: { "x-deerflow-desktop-internal-next": "internal-secret" },
    });

    expect(await res.json()).toEqual({
      url: "/api/v1/auth/setup-status",
      token: "secret-token",
    });
  });
});
```

- [ ] **Step 2: Implement the proxy**

Create `desktop/electron/src/proxy/desktop-proxy.ts`:

```ts
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { appendFile, readFile } from "node:fs/promises";
import httpProxy from "http-proxy";

const INTERNAL_NEXT_HEADER = "x-deerflow-desktop-internal-next";
type HttpProxyServer = ReturnType<typeof httpProxy.createProxyServer>;

export interface DesktopProxyOptions {
  host: string;
  port: number;
  gatewayOrigin: string;
  nextOrigin: string;
  tokenPath: string;
  internalHeaderValue: string;
  logPath: string;
}

export interface DesktopProxyServer {
  origin: string;
  close: () => Promise<void>;
}

export async function startDesktopProxy(options: DesktopProxyOptions): Promise<DesktopProxyServer> {
  const proxy = httpProxy.createProxyServer({ changeOrigin: true, xfwd: true });
  const server = createServer(async (req, res) => {
    try {
      if (shouldProxyGateway(req)) {
        await proxyGateway(req, res, proxy, options, req.url ?? "/");
        return;
      }

      if (shouldProxyInternalGateway(req, options.internalHeaderValue)) {
        const targetUrl = (req.url ?? "/").replace(/^\/_desktop-gateway/, "") || "/";
        await proxyGateway(req, res, proxy, options, targetUrl);
        return;
      }

      if ((req.url ?? "/").startsWith("/_desktop-gateway/")) {
        res.statusCode = 404;
        res.end("Not found");
        return;
      }

      proxy.web(req, res, { target: options.nextOrigin });
    } catch (err) {
      await logProxyError(options.logPath, err);
      res.statusCode = 502;
      res.end(`Desktop proxy error: ${String(err)}`);
    }
  });

  await new Promise<void>((resolve) => server.listen(options.port, options.host, () => resolve()));
  return {
    origin: `http://${options.host}:${options.port}`,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

function shouldProxyGateway(req: IncomingMessage): boolean {
  const url = req.url ?? "/";
  return url === "/api" || url.startsWith("/api/");
}

function shouldProxyInternalGateway(req: IncomingMessage, internalHeaderValue: string): boolean {
  const url = req.url ?? "/";
  if (!url.startsWith("/_desktop-gateway/")) return false;
  return req.headers[INTERNAL_NEXT_HEADER] === internalHeaderValue;
}

async function proxyGateway(
  req: IncomingMessage,
  res: ServerResponse,
  proxy: HttpProxyServer,
  options: DesktopProxyOptions,
  targetUrl: string,
): Promise<void> {
  const token = (await readFile(options.tokenPath, "utf-8")).trim();
  req.headers["x-deerflow-desktop-token"] = token;
  req.url = mapGatewayUrl(targetUrl);
  proxy.web(req, res, { target: options.gatewayOrigin });
}

function mapGatewayUrl(url: string): string {
  if (url === "/api/langgraph") return "/api";
  if (url.startsWith("/api/langgraph?")) return `/api${url.slice("/api/langgraph".length)}`;
  if (url.startsWith("/api/langgraph/")) {
    return `/api/${url.slice("/api/langgraph/".length)}`;
  }
  return url;
}

async function logProxyError(logPath: string, err: unknown): Promise<void> {
  await appendFile(logPath, `[${new Date().toISOString()}] ${String(err)}\n`, "utf-8");
}
```

- [ ] **Step 3: Add the proxy bundle entry**

Modify `desktop/electron/tsup.config.ts`:

```ts
export default defineConfig({
  entry: {
    "main/index": "src/main/index.ts",
    "preload/index": "src/preload/index.ts",
    "proxy/desktop-proxy": "src/proxy/desktop-proxy.ts",
  },
  // keep the existing remaining options
});
```

- [ ] **Step 4: Run proxy tests**

Run: `cd desktop/electron && pnpm test -- tests/proxy.test.ts`

Expected: PASS.

- [ ] **Step 5: Run all desktop unit tests and build**

Run:

```bash
cd desktop/electron
pnpm test
pnpm build
```

Expected: PASS and `dist/proxy/desktop-proxy.js` exists.

- [ ] **Step 6: Commit**

```bash
git add desktop/electron/src/proxy/desktop-proxy.ts desktop/electron/tests/proxy.test.ts desktop/electron/tsup.config.ts
git commit -m "feat: add desktop gateway proxy"
```

---

### Task 6: Runtime Orchestration for Dev and Packaged Modes

Wire app-data, env, ports, sidecars, proxy, and BrowserWindow together. Developer mode may use local `uv`/`pnpm`; packaged mode must use resources.

**Files:**
- Create: `desktop/electron/src/main/runtime.ts`
- Create: `desktop/electron/src/main/window.ts`
- Modify: `desktop/electron/src/main/index.ts`
- Modify: `desktop/electron/src/preload/index.ts`
- Create: `desktop/electron/src/next/register-fetch.ts`
- Modify: `desktop/electron/tsup.config.ts`
- Create: `desktop/electron/tests/runtime.test.ts`

- [ ] **Step 1: Write failing runtime command tests**

Create `desktop/electron/tests/runtime.test.ts`:

```ts
import { describe, expect, test } from "vitest";

import { buildGatewayCommand, buildNextCommand, classifyReadinessFailure } from "../src/main/runtime";
import { resolveDesktopResources } from "../src/main/paths";

describe("runtime command builders", () => {
  test("builds dev gateway command using uvicorn on 127.0.0.1", () => {
    const command = buildGatewayCommand({
      packaged: false,
      backendDir: "/repo/backend",
      pythonBin: "python",
      port: 41234,
      dataRoot: "/data",
    });

    expect(command.cwd).toBe("/data");
    expect(command.command).toBe("uv");
    expect(command.args).toContain("--project");
    expect(command.args).toContain("/repo/backend");
    expect(command.env.PYTHONPATH).toContain("/repo/backend");
    expect(command.args).toContain("uvicorn");
    expect(command.args).toContain("app.gateway.app:app");
    expect(command.args).toContain("127.0.0.1");
  });

  test("builds packaged next standalone command", () => {
    const command = buildNextCommand({
      packaged: true,
      frontendDir: "/resources/frontend",
      nodeBin: "/resources/runtimes/node/bin/node",
      port: 42345,
      registerFetchPath: "/resources/desktop-server/next/register-fetch.js",
    });

    expect(command.command).toMatch(/[/\\]resources[/\\]runtimes[/\\]node[/\\]bin[/\\]node$/);
    expect(command.args[0]).toBe("--import");
    expect(command.args[1]).toMatch(/[/\\]resources[/\\]desktop-server[/\\]next[/\\]register-fetch\.js$/);
    expect(command.args[2]).toMatch(/[/\\]resources[/\\]frontend[/\\]\.next[/\\]standalone[/\\]server\.js$/);
    expect(command.env.PORT).toBe("42345");
    expect(command.env.HOSTNAME).toBe("127.0.0.1");
  });

  test("builds dev next command with SSR fetch marker", () => {
    const command = buildNextCommand({
      packaged: false,
      frontendDir: "/repo/frontend",
      nodeBin: "node",
      port: 43333,
      registerFetchPath: "/repo/desktop/electron/dist/next/register-fetch.js",
    });

    expect(command.command).toBe("pnpm");
    expect(command.args).toEqual(["dev", "--hostname", "127.0.0.1", "--port", "43333"]);
    expect(command.env.NODE_OPTIONS).toContain("--import /repo/desktop/electron/dist/next/register-fetch.js");
  });

  test("resolves dev desktop server bundle directory", () => {
    const resources = resolveDesktopResources({
      packaged: false,
      appPath: "/repo/desktop/electron",
      resourcesPath: "/ignored",
    });

    expect(resources.desktopServerDir).toMatch(/[/\\]repo[/\\]desktop[/\\]electron[/\\]dist$/);
  });

  test("classifies startup failures with actionable diagnostics", () => {
    expect(classifyReadinessFailure("gateway", "timeout", "ModuleNotFoundError: No module named 'app'")).toContain("Gateway failed to import");
    expect(classifyReadinessFailure("gateway", "timeout", "Failed to load configuration during gateway startup")).toContain("config invalid");
    expect(classifyReadinessFailure("gateway", "timeout", "No models configured")).toContain("model key missing");
    expect(classifyReadinessFailure("frontend", "timeout", "next dev failed")).toContain("frontend failed to start");
    expect(classifyReadinessFailure("gateway", "timeout", "LocalSandboxProvider allow_host_bash")).toContain("local sandbox configuration invalid");
  });
});
```

- [ ] **Step 2: Implement runtime command builders and launcher**

Create `desktop/electron/src/main/runtime.ts`:

```ts
import { randomBytes } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join, delimiter } from "node:path";

import { ensureDesktopData, type DesktopDataPaths } from "./app-data";
import { allocateDesktopPorts } from "./ports";
import { buildGatewayEnv, buildNextEnv } from "./env";
import { resolveDesktopResources, type DesktopResources } from "./paths";
import { startSidecar, type SidecarProcess } from "./sidecar";
import { startDesktopProxy, type DesktopProxyServer } from "../proxy/desktop-proxy";

export interface RuntimeCommand {
  command: string;
  args: string[];
  cwd: string;
  env: NodeJS.ProcessEnv;
}

export const INTERNAL_NEXT_HEADER = "x-deerflow-desktop-internal-next";

function buildPythonPath(paths: string[]): string {
  return paths.filter(Boolean).join(delimiter);
}

export function buildGatewayCommand(options: {
  packaged: boolean;
  backendDir: string;
  pythonBin: string;
  port: number;
  dataRoot: string;
}): RuntimeCommand {
  if (!options.packaged) {
    return {
      command: "uv",
      args: ["--project", options.backendDir, "run", "uvicorn", "app.gateway.app:app", "--host", "127.0.0.1", "--port", String(options.port)],
      cwd: options.dataRoot,
      env: {
        GATEWAY_PORT: String(options.port),
        PYTHONPATH: buildPythonPath([
          options.backendDir,
          join(options.backendDir, "packages", "harness"),
          process.env.PYTHONPATH ?? "",
        ]),
      },
    };
  }

  const pythonPath = buildPythonPath([
    options.backendDir,
    join(options.backendDir, "packages", "harness"),
    join(options.backendDir, "site-packages"),
  ]);

  return {
    command: options.pythonBin,
    args: ["-m", "uvicorn", "app.gateway.app:app", "--host", "127.0.0.1", "--port", String(options.port)],
    cwd: options.dataRoot,
    env: { GATEWAY_PORT: String(options.port), PYTHONPATH: pythonPath },
  };
}

export function buildNextCommand(options: {
  packaged: boolean;
  frontendDir: string;
  nodeBin: string;
  port: number;
  registerFetchPath?: string;
}): RuntimeCommand {
  if (!options.packaged) {
    const importOption = options.registerFetchPath ? `--import ${options.registerFetchPath}` : "";
    const nodeOptions = [process.env.NODE_OPTIONS, importOption].filter(Boolean).join(" ");
    return {
      command: "pnpm",
      args: ["dev", "--hostname", "127.0.0.1", "--port", String(options.port)],
      cwd: options.frontendDir,
      env: { NODE_OPTIONS: nodeOptions, PORT: String(options.port), HOSTNAME: "127.0.0.1" },
    };
  }

  return {
    command: options.nodeBin,
    args: [
      ...(options.registerFetchPath ? ["--import", options.registerFetchPath] : []),
      join(options.frontendDir, ".next", "standalone", "server.js"),
    ],
    cwd: options.frontendDir,
    env: { PORT: String(options.port), HOSTNAME: "127.0.0.1" },
  };
}

export interface DesktopRuntime {
  proxyOrigin: string;
  stop: () => Promise<void>;
}

export async function waitForHttpReady(url: string, name: string, logPath: string, timeoutMs = 30_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let lastError = "";
  while (Date.now() < deadline) {
    try {
      const res = await fetch(url);
      if (res.ok || res.status < 500) return;
      lastError = `${res.status} ${res.statusText}`;
    } catch (err) {
      lastError = String(err);
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(classifyReadinessFailure(name, lastError, await readLogTail(logPath)));
}

async function readLogTail(path: string): Promise<string> {
  try {
    return (await readFile(path, "utf-8")).slice(-8000);
  } catch {
    return "";
  }
}

export function classifyReadinessFailure(name: string, lastError: string, logTail = ""): string {
  const text = `${lastError}\n${logTail}`;
  if (/ModuleNotFoundError|ImportError|Could not import module/i.test(text)) {
    return `Gateway failed to import. Check the ${name} log in the desktop app-data logs directory. Last error: ${lastError}`;
  }
  if (/Failed to load configuration|ValidationError|YAML|config/i.test(text)) {
    return `Gateway config invalid. Check ${name}.log and the desktop config.yaml. Last error: ${lastError}`;
  }
  if (/No models configured|api[_ -]?key|missing model/i.test(text)) {
    return `Gateway model key missing or no model configured. Open desktop settings and update app-data .env/config.yaml. Last error: ${lastError}`;
  }
  if (/LocalSandboxProvider|allow_host_bash|sandbox/i.test(text)) {
    return `Gateway local sandbox configuration invalid. Desktop supports only LocalSandboxProvider with allow_host_bash false by default. Last error: ${lastError}`;
  }
  if (/next|frontend|EADDRINUSE|Failed to start/i.test(text) || name === "frontend") {
    return `Desktop frontend failed to start. Check frontend.log in the desktop app-data logs directory. Last error: ${lastError}`;
  }
  return `${name} did not become ready before timeout. Last error: ${lastError}. Check the ${name} log in the desktop app-data logs directory.`;
}

export async function startDesktopRuntime(options: {
  appDataRoot: string;
  packaged: boolean;
  appPath: string;
  resourcesPath: string;
}): Promise<DesktopRuntime> {
  const resources = resolveDesktopResources(options);
  const data = await ensureDesktopData({
    root: options.appDataRoot,
    exampleConfigPath: resources.repoRoot
      ? join(resources.repoRoot, "config.example.yaml")
      : join(resources.backendDir, "config.example.yaml"),
    exampleExtensionsConfigPath: resources.repoRoot
      ? join(resources.repoRoot, "extensions_config.example.json")
      : join(resources.backendDir, "extensions_config.example.json"),
  });
  const ports = await allocateDesktopPorts();
  const proxyOrigin = `http://127.0.0.1:${ports.proxyPort}`;
  const internalHeaderValue = randomBytes(32).toString("base64url");
  const sidecars: SidecarProcess[] = [];
  let proxy: DesktopProxyServer | null = null;

  try {
    const gateway = await startGateway(data, resources, options.packaged, ports.gatewayPort, proxyOrigin);
    sidecars.push(gateway);
    await waitForHttpReady(`http://127.0.0.1:${ports.gatewayPort}/health`, "gateway", join(data.logsDir, "gateway.log"));

    proxy = await startDesktopProxy({
      host: "127.0.0.1",
      port: ports.proxyPort,
      gatewayOrigin: `http://127.0.0.1:${ports.gatewayPort}`,
      nextOrigin: `http://127.0.0.1:${ports.nextPort}`,
      tokenPath: data.tokenPath,
      internalHeaderValue,
      logPath: join(data.logsDir, "desktop-proxy.log"),
    });

    const next = await startNext(data, resources, options.packaged, ports.nextPort, proxyOrigin, internalHeaderValue);
    sidecars.push(next);
    await waitForHttpReady(proxy.origin, "frontend", join(data.logsDir, "frontend.log"));
  } catch (err) {
    if (proxy) await proxy.close();
    await Promise.allSettled(sidecars.map((sidecar) => sidecar.stop()));
    throw err;
  }

  return {
    proxyOrigin,
    stop: async () => {
      if (proxy) await proxy.close();
      await Promise.allSettled(sidecars.reverse().map((sidecar) => sidecar.stop()));
    },
  };
}

async function startGateway(
  data: DesktopDataPaths,
  resources: DesktopResources,
  packaged: boolean,
  port: number,
  proxyOrigin: string,
): Promise<SidecarProcess> {
  const command = buildGatewayCommand({
    packaged,
    backendDir: resources.backendDir,
    pythonBin: resources.pythonBin,
    port,
    dataRoot: data.root,
  });
  command.env = { ...(await buildGatewayEnv(data, { frontendOrigin: proxyOrigin })), ...command.env };
  return startSidecar({
    name: "gateway",
    command: command.command,
    args: command.args,
    cwd: command.cwd,
    env: command.env,
    logPath: join(data.logsDir, "gateway.log"),
  });
}

async function startNext(
  data: DesktopDataPaths,
  resources: DesktopResources,
  packaged: boolean,
  port: number,
  proxyOrigin: string,
  internalHeaderValue: string,
): Promise<SidecarProcess> {
  const command = buildNextCommand({
    packaged,
    frontendDir: resources.frontendDir,
    nodeBin: resources.nodeBin,
    port,
    registerFetchPath: join(resources.desktopServerDir, "next", "register-fetch.js"),
  });
  command.env = { ...(await buildNextEnv(data, { proxyOrigin })), ...command.env };
  command.env.DEER_FLOW_DESKTOP_INTERNAL_NEXT_HEADER = INTERNAL_NEXT_HEADER;
  command.env.DEER_FLOW_DESKTOP_INTERNAL_NEXT_VALUE = internalHeaderValue;
  return startSidecar({
    name: "frontend",
    command: command.command,
    args: command.args,
    cwd: command.cwd,
    env: command.env,
    logPath: join(data.logsDir, "frontend.log"),
  });
}
```

- [ ] **Step 3: Add Next SSR fetch marker**

Create `desktop/electron/src/next/register-fetch.ts`:

```ts
const internalHeader = process.env.DEER_FLOW_DESKTOP_INTERNAL_NEXT_HEADER;
const internalHeaderValue = process.env.DEER_FLOW_DESKTOP_INTERNAL_NEXT_VALUE;
const gatewayBase = process.env.DEER_FLOW_INTERNAL_GATEWAY_BASE_URL;
const originalFetch = globalThis.fetch;

if (internalHeader && internalHeaderValue && gatewayBase) {
  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === "string" || input instanceof URL ? String(input) : input.url;
    if (url.startsWith(gatewayBase)) {
      const headers = new Headers(init?.headers ?? (typeof input === "object" && "headers" in input ? input.headers : undefined));
      headers.set(internalHeader, internalHeaderValue);
      return originalFetch(input, { ...init, headers });
    }
    return originalFetch(input, init);
  };
}
```

- [ ] **Step 4: Add runtime bundle entries**

Modify `desktop/electron/tsup.config.ts`:

```ts
export default defineConfig({
  entry: {
    "main/index": "src/main/index.ts",
    "preload/index": "src/preload/index.ts",
    "proxy/desktop-proxy": "src/proxy/desktop-proxy.ts",
    "next/register-fetch": "src/next/register-fetch.ts",
  },
  // keep the existing remaining options
});
```

- [ ] **Step 5: Add BrowserWindow factory**

Create `desktop/electron/src/main/window.ts`:

```ts
import { BrowserWindow, shell } from "electron";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

export async function createMainWindow(url: string): Promise<BrowserWindow> {
  const win = new BrowserWindow({
    width: 1280,
    height: 840,
    minWidth: 960,
    minHeight: 640,
    webPreferences: {
      preload: join(__dirname, "..", "preload", "index.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  win.webContents.setWindowOpenHandler(({ url: target }) => {
    void shell.openExternal(target);
    return { action: "deny" };
  });

  await win.loadURL(url);
  return win;
}
```

- [ ] **Step 6: Wire Electron lifecycle**

Modify `desktop/electron/src/main/index.ts`:

```ts
import { app } from "electron";

import { startDesktopRuntime, type DesktopRuntime } from "./runtime";
import { createMainWindow } from "./window";

let runtime: DesktopRuntime | null = null;
let quitting = false;

async function main() {
  await app.whenReady();
  if (process.env.DEER_FLOW_DESKTOP_SMOKE_URL) {
    await createMainWindow(process.env.DEER_FLOW_DESKTOP_SMOKE_URL);
    return;
  }
  runtime = await startDesktopRuntime({
    appDataRoot: app.getPath("userData"),
    packaged: app.isPackaged,
    appPath: app.getAppPath(),
    resourcesPath: process.resourcesPath,
  });
  await createMainWindow(runtime.proxyOrigin);
}

app.on("before-quit", async (event) => {
  if (quitting || !runtime) {
    return;
  }
  event.preventDefault();
  quitting = true;
  const current = runtime;
  runtime = null;
  await current.stop();
  app.exit(0);
});

void main().catch((err) => {
  console.error("[desktop] failed to start", err);
  app.exit(1);
});
```

Modify `desktop/electron/src/preload/index.ts`:

```ts
import { contextBridge } from "electron";

contextBridge.exposeInMainWorld("deerFlowDesktop", {
  platform: process.platform,
});
```

- [ ] **Step 7: Run runtime tests and typecheck**

Run:

```bash
cd desktop/electron
pnpm test -- tests/runtime.test.ts
pnpm build
```

Expected: tests PASS and `tsc` exits 0.

- [ ] **Step 8: Commit**

```bash
git add desktop/electron/src/main/runtime.ts desktop/electron/src/main/window.ts desktop/electron/src/main/index.ts desktop/electron/src/preload/index.ts desktop/electron/src/next/register-fetch.ts desktop/electron/tests/runtime.test.ts desktop/electron/tsup.config.ts
git commit -m "feat: orchestrate desktop local runtime"
```

---

### Task 7: Packaged Runtime Staging Scripts

Add scripts that stage existing backend/frontend artifacts into local `desktop/electron/resources/` without changing core build scripts. `desktop/electron/resources/` is generated, ignored by git, and must not be committed. The MVP staging script accepts prebuilt Python and Node runtime paths; CI/release packaging is responsible for producing those relocatable runtime folders from lockfiles before this script runs.

**Files:**
- Create: `desktop/electron/src/scripts/build-runtime.ts`
- Create: `desktop/electron/src/scripts/stage-python-deps.ts`
- Create: `desktop/electron/src/scripts/smoke-packaged.ts`
- Create: `desktop/electron/src/scripts/smoke-runtime.ts`
- Modify: `desktop/electron/tsup.config.ts`
- Create: `desktop/electron/tests/build-runtime.test.ts`

- [ ] **Step 1: Write failing staging layout tests**

Create `desktop/electron/tests/build-runtime.test.ts`:

```ts
import { describe, expect, test } from "vitest";

import { runtimeLayout } from "../src/scripts/build-runtime";

describe("runtimeLayout", () => {
  test("defines backend, frontend, runtimes, and desktop-server resources", () => {
    const layout = runtimeLayout("/repo", "/out");

    expect(layout.backend).toMatch(/[/\\]out[/\\]backend$/);
    expect(layout.frontend).toMatch(/[/\\]out[/\\]frontend$/);
    expect(layout.pythonRuntime).toMatch(/[/\\]out[/\\]runtimes[/\\]python$/);
    expect(layout.nodeRuntime).toMatch(/[/\\]out[/\\]runtimes[/\\]node$/);
    expect(layout.desktopServer).toMatch(/[/\\]out[/\\]desktop-server$/);
  });

  test("does not define non-local sandbox resources", () => {
    const layout = runtimeLayout("/repo", "/out");

    expect(Object.values(layout).join("\n")).not.toMatch(/docker|aio|apple container|container_prefix/i);
  });
});
```

- [ ] **Step 2: Implement runtime staging script**

Create `desktop/electron/src/scripts/build-runtime.ts`:

```ts
import { cp, mkdir, rm } from "node:fs/promises";
import { dirname, join } from "node:path";

export interface RuntimeLayout {
  backend: string;
  frontend: string;
  pythonRuntime: string;
  nodeRuntime: string;
  desktopServer: string;
}

export function runtimeLayout(_repoRoot: string, outDir: string): RuntimeLayout {
  return {
    backend: join(outDir, "backend"),
    frontend: join(outDir, "frontend"),
    pythonRuntime: join(outDir, "runtimes", "python"),
    nodeRuntime: join(outDir, "runtimes", "node"),
    desktopServer: join(outDir, "desktop-server"),
  };
}

export async function stageRuntime(repoRoot: string, outDir: string): Promise<void> {
  const layout = runtimeLayout(repoRoot, outDir);
  await rm(outDir, { recursive: true, force: true });
  await mkdir(outDir, { recursive: true });

  await cp(join(repoRoot, "backend", "app"), join(layout.backend, "app"), { recursive: true });
  await cp(join(repoRoot, "backend", "packages"), join(layout.backend, "packages"), { recursive: true });
  await cp(join(repoRoot, "backend", "pyproject.toml"), join(layout.backend, "pyproject.toml"));
  await cp(join(repoRoot, "backend", "uv.lock"), join(layout.backend, "uv.lock"));
  await cp(join(repoRoot, "config.example.yaml"), join(layout.backend, "config.example.yaml"));
  await cp(join(repoRoot, "extensions_config.example.json"), join(layout.backend, "extensions_config.example.json"));
  await stagePythonDeps(join(layout.backend, "site-packages"));

  await cp(join(repoRoot, "frontend", ".next", "standalone"), join(layout.frontend, ".next", "standalone"), { recursive: true });
  await cp(join(repoRoot, "frontend", ".next", "static"), join(layout.frontend, ".next", "standalone", ".next", "static"), { recursive: true });
  await cp(join(repoRoot, "frontend", "public"), join(layout.frontend, ".next", "standalone", "public"), { recursive: true });

  await cp(join(repoRoot, "desktop", "electron", "dist", "proxy"), join(layout.desktopServer, "proxy"), { recursive: true });
  await cp(join(repoRoot, "desktop", "electron", "dist", "next"), join(layout.desktopServer, "next"), { recursive: true });

  await stageRuntimeFolder(process.env.DEER_FLOW_DESKTOP_NODE_RUNTIME_DIR, layout.nodeRuntime, "DEER_FLOW_DESKTOP_NODE_RUNTIME_DIR");
  await stageRuntimeFolder(process.env.DEER_FLOW_DESKTOP_PYTHON_RUNTIME_DIR, layout.pythonRuntime, "DEER_FLOW_DESKTOP_PYTHON_RUNTIME_DIR");
}

async function stagePythonDeps(destination: string): Promise<void> {
  const source = process.env.DEER_FLOW_DESKTOP_PYTHON_SITE_PACKAGES_DIR;
  if (!source) {
    throw new Error("DEER_FLOW_DESKTOP_PYTHON_SITE_PACKAGES_DIR must point to wheel-installed Python dependencies for this platform");
  }
  await rm(destination, { recursive: true, force: true });
  await mkdir(dirname(destination), { recursive: true });
  await cp(source, destination, { recursive: true });
}

async function stageRuntimeFolder(source: string | undefined, destination: string, envName: string): Promise<void> {
  if (!source) {
    throw new Error(`${envName} must point to a prebuilt relocatable runtime folder`);
  }
  await rm(destination, { recursive: true, force: true });
  await mkdir(dirname(destination), { recursive: true });
  await cp(source, destination, { recursive: true });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const repoRoot = process.argv[2];
  const outDir = process.argv[3] ?? join(process.cwd(), "resources");
  if (!repoRoot) {
    throw new Error("Usage: node dist/scripts/build-runtime.js <repo-root> [out-dir]");
  }
  await stageRuntime(repoRoot, outDir);
}
```

- [ ] **Step 3: Add Python dependency staging helper**

Create `desktop/electron/src/scripts/stage-python-deps.ts`:

```ts
import { spawnSync } from "node:child_process";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

async function main() {
  const backendDir = process.argv[2] ?? join(process.cwd(), "..", "..", "backend");
  const outDir = process.argv[3];
  if (!outDir) {
    throw new Error("Usage: node dist/scripts/stage-python-deps.js <backend-dir> <out-site-packages-dir>");
  }

  await mkdir(outDir, { recursive: true });
  const tempDir = await mkdtemp(join(tmpdir(), "deerflow-desktop-reqs-"));
  const requirementsPath = join(tempDir, "requirements.txt");
  const exportResult = spawnSync(
    "uv",
    [
      "export",
      "--frozen",
      "--all-packages",
      "--no-dev",
      "--no-emit-project",
      "--no-emit-workspace",
      "--format",
      "requirements.txt",
      "--output-file",
      requirementsPath,
    ],
    { cwd: backendDir, stdio: "inherit" },
  );
  if (exportResult.status !== 0) {
    await rm(tempDir, { recursive: true, force: true });
    throw new Error(`uv export failed with status ${exportResult.status}`);
  }

  const result = spawnSync(
    "uv",
    ["pip", "install", "--python", process.env.DEER_FLOW_DESKTOP_BUILD_PYTHON ?? "python", "--target", outDir, "-r", requirementsPath],
    { cwd: backendDir, stdio: "inherit" },
  );
  await rm(tempDir, { recursive: true, force: true });
  if (result.status !== 0) {
    throw new Error(`uv pip install failed with status ${result.status}`);
  }
}

await main();
```

This helper is for CI/release build machines only. It installs from the locked backend project into a target directory that is later copied into `resources/backend/site-packages`; it must not run on customer machines.

- [ ] **Step 4: Add packaged smoke helper**

Create `desktop/electron/src/scripts/smoke-packaged.ts`:

```ts
import { access } from "node:fs/promises";
import { join } from "node:path";

async function main() {
  const resources = process.argv[2] ?? join(process.cwd(), "resources");
  const required = [
    join(resources, "backend", "app"),
    join(resources, "backend", "packages"),
    join(resources, "backend", "config.example.yaml"),
    join(resources, "backend", "extensions_config.example.json"),
    join(resources, "backend", "site-packages"),
    join(resources, "frontend", ".next", "standalone", "server.js"),
    join(resources, "frontend", ".next", "standalone", ".next", "static"),
    join(resources, "frontend", ".next", "standalone", "public"),
    join(resources, "desktop-server", "proxy"),
    join(resources, "desktop-server", "next", "register-fetch.js"),
    process.platform === "win32" ? join(resources, "runtimes", "node", "node.exe") : join(resources, "runtimes", "node", "bin", "node"),
    process.platform === "win32" ? join(resources, "runtimes", "python", "python.exe") : join(resources, "runtimes", "python", "bin", "python"),
  ];

  for (const path of required) {
    await access(path);
  }

  console.log("Packaged runtime layout smoke passed");
}

await main();
```

- [ ] **Step 5: Add real desktop runtime smoke helper**

Create `desktop/electron/src/scripts/smoke-runtime.ts`:

```ts
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { startDesktopRuntime } from "../main/runtime";

function cookieHeader(cookies: Map<string, string>): string {
  return [...cookies.entries()].map(([key, value]) => `${key}=${value}`).join("; ");
}

function rememberSetCookie(cookies: Map<string, string>, response: Response) {
  const headers = response.headers as Headers & { getSetCookie?: () => string[] };
  const raw = headers.getSetCookie?.() ?? [];
  const fallback = response.headers.get("set-cookie");
  const values = raw.length > 0 ? raw : fallback ? [fallback] : [];
  for (const value of values) {
    const [pair] = value.split(";");
    const [name, cookieValue] = pair.split("=", 2);
    if (name && cookieValue) cookies.set(name.trim(), cookieValue.trim());
  }
}

async function requestJson(url: string, init: RequestInit = {}) {
  const response = await fetch(url, init);
  const text = await response.text();
  let body: unknown = text;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    // keep text body
  }
  return { response, body };
}

async function main() {
  const repoRoot = process.argv[2] ?? join(process.cwd(), "..", "..");
  const packaged = process.argv.includes("--packaged");
  const resourcesPath = process.argv.includes("--resources")
    ? process.argv[process.argv.indexOf("--resources") + 1]
    : join(process.cwd(), "resources");
  const appPath = packaged ? process.cwd() : join(repoRoot, "desktop", "electron");
  const appDataRoot = await mkdtemp(join(tmpdir(), "deerflow desktop smoke "));
  const runtime = await startDesktopRuntime({
    appDataRoot,
    packaged,
    appPath,
    resourcesPath,
  });
  const cookies = new Map<string, string>();

  try {
    const setup = await requestJson(`${runtime.proxyOrigin}/api/v1/auth/setup-status`);
    if (!setup.response.ok) throw new Error(`setup-status failed: ${setup.response.status} ${JSON.stringify(setup.body)}`);

    const init = await requestJson(`${runtime.proxyOrigin}/api/v1/auth/initialize`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ email: "desktop-smoke@example.com", password: "Str0ng!Pass99" }),
    });
    if (init.response.status !== 201) {
      throw new Error(`initialize failed: ${init.response.status} ${JSON.stringify(init.body)}`);
    }
    rememberSetCookie(cookies, init.response);

    const csrf = cookies.get("csrf_token");
    const models = await requestJson(`${runtime.proxyOrigin}/api/models`, {
      headers: {
        Cookie: cookieHeader(cookies),
        ...(csrf ? { "x-csrf-token": csrf } : {}),
      },
    });
    if (!models.response.ok || typeof models.body !== "object" || models.body === null || !("models" in models.body)) {
      throw new Error(`models smoke failed: ${models.response.status} ${JSON.stringify(models.body)}`);
    }

    const config = await readFile(join(appDataRoot, "config.yaml"), "utf-8");
    if (!config.includes("deerflow.sandbox.local:LocalSandboxProvider") || config.includes("container_prefix")) {
      throw new Error("desktop config is not local-sandbox only");
    }

    console.log(`Desktop runtime smoke passed (${packaged ? "packaged" : "dev"} mode)`);
  } finally {
    await runtime.stop();
    await rm(appDataRoot, { recursive: true, force: true });
  }
}

await main();
```

This smoke intentionally uses an app-data directory whose path contains spaces. It verifies the real Gateway + Next + desktop proxy path, desktop token injection, existing auth/session flow, `/api/models`, and generated `LocalSandboxProvider` config. It does not require model API keys because `/api/models` may validly return an empty `models` list.

- [ ] **Step 6: Add packaged runtime script bundle entries**

Modify `desktop/electron/tsup.config.ts`:

```ts
export default defineConfig({
  entry: {
    "main/index": "src/main/index.ts",
    "preload/index": "src/preload/index.ts",
    "proxy/desktop-proxy": "src/proxy/desktop-proxy.ts",
    "next/register-fetch": "src/next/register-fetch.ts",
    "scripts/build-runtime": "src/scripts/build-runtime.ts",
    "scripts/smoke-packaged": "src/scripts/smoke-packaged.ts",
    "scripts/smoke-runtime": "src/scripts/smoke-runtime.ts",
    "scripts/stage-python-deps": "src/scripts/stage-python-deps.ts",
  },
  // keep the existing remaining options
});
```

- [ ] **Step 7: Build frontend standalone**

Run:

```bash
cd frontend
NEXT_CONFIG_BUILD_OUTPUT=standalone SKIP_ENV_VALIDATION=1 pnpm build
```

Expected: `.next/standalone/server.js` exists.

- [ ] **Step 8: Run staging tests and script**

Before running the script, set these operator-supplied paths for the current platform:

```bash
export DEER_FLOW_DESKTOP_BUILD_PYTHON="$DESKTOP_PYTHON_RUNTIME_DIR/bin/python"
export DEER_FLOW_DESKTOP_NODE_RUNTIME_DIR="$DESKTOP_NODE_RUNTIME_DIR"
export DEER_FLOW_DESKTOP_PYTHON_RUNTIME_DIR="$DESKTOP_PYTHON_RUNTIME_DIR"
export DEER_FLOW_DESKTOP_PYTHON_SITE_PACKAGES_DIR="/tmp/deerflow-desktop-site-packages"
```

On Windows build hosts, use the Python runtime's `python.exe` path for `DEER_FLOW_DESKTOP_BUILD_PYTHON`.

Run:

```bash
cd desktop/electron
pnpm test -- tests/build-runtime.test.ts
pnpm build
node dist/scripts/stage-python-deps.js ../../backend "$DEER_FLOW_DESKTOP_PYTHON_SITE_PACKAGES_DIR"
node dist/scripts/build-runtime.js ../.. resources
pnpm smoke resources
node dist/scripts/smoke-runtime.js ../.. --packaged --resources resources
```

Expected: tests PASS, layout smoke prints `Packaged runtime layout smoke passed`, and real runtime smoke prints `Desktop runtime smoke passed (packaged mode)`. The two runtime env vars must point at platform-specific, prebuilt relocatable runtime folders; do not run dependency installation on the customer machine.

- [ ] **Step 9: Commit**

```bash
git add desktop/electron/src/scripts/build-runtime.ts desktop/electron/src/scripts/stage-python-deps.ts desktop/electron/src/scripts/smoke-packaged.ts desktop/electron/src/scripts/smoke-runtime.ts desktop/electron/tests/build-runtime.test.ts desktop/electron/tsup.config.ts
git commit -m "feat: stage desktop packaged runtime"
```

---

### Task 8: Electron Smoke Test and Manual Cross-Platform Matrix

Add a launch smoke test that verifies the Electron app can open a local shell window with a stub runtime. Do not require real model keys.

**Files:**
- Create: `desktop/electron/tests/electron-smoke.spec.ts`
- Modify: `desktop/electron/package.json`
- Create: `desktop/electron/tests/fixtures/stub-runtime.ts`

- [ ] **Step 1: Add Electron smoke dependencies and script**

Modify `desktop/electron/package.json`:

```json
{
  "scripts": {
    "test:electron": "playwright test tests/electron-smoke.spec.ts"
  },
  "devDependencies": {
    "@playwright/test": "^1.59.1"
  }
}
```

Keep existing scripts/dependencies intact.

- [ ] **Step 2: Add smoke fixture**

Create `desktop/electron/tests/fixtures/stub-runtime.ts`:

```ts
import { createServer } from "node:http";

export async function startStubFrontend(): Promise<{ url: string; close: () => Promise<void> }> {
  const server = createServer((_req, res) => {
    res.setHeader("content-type", "text/html");
    res.end("<html><body><main>DeerFlow Desktop Smoke</main></body></html>");
  });

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("missing stub address");

  return {
    url: `http://127.0.0.1:${address.port}`,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}
```

- [ ] **Step 3: Add Electron smoke test**

Create `desktop/electron/tests/electron-smoke.spec.ts`:

```ts
import { test, expect, _electron as electron } from "@playwright/test";
import { join } from "node:path";

import { startStubFrontend } from "./fixtures/stub-runtime";

test("desktop window can render local runtime origin", async () => {
  const stub = await startStubFrontend();
  const app = await electron.launch({
    args: [join(process.cwd(), "dist", "main", "index.js")],
    env: {
      ...process.env,
      DEER_FLOW_DESKTOP_SMOKE_URL: stub.url,
    },
  });

  const page = await app.firstWindow();
  await expect(page.locator("main")).toContainText("DeerFlow Desktop Smoke");

  await app.close();
  await stub.close();
});
```

- [ ] **Step 4: Confirm smoke override in `index.ts`**

Confirm `desktop/electron/src/main/index.ts` already contains this branch inside `main()` before `startDesktopRuntime(...)` from Task 6:

```ts
  if (process.env.DEER_FLOW_DESKTOP_SMOKE_URL) {
    await createMainWindow(process.env.DEER_FLOW_DESKTOP_SMOKE_URL);
    return;
  }
```

- [ ] **Step 5: Run Electron smoke test**

Run:

```bash
cd desktop/electron
pnpm build
pnpm test:electron
```

Expected: PASS with a rendered `DeerFlow Desktop Smoke` page.

- [ ] **Step 6: Record manual matrix in docs**

Add the manual matrix to `docs/desktop-local-runtime.md` in Task 9. At this point only mark local machine smoke as run; leave Windows/macOS installer rows unchecked until actually run.

- [ ] **Step 7: Commit**

```bash
git add desktop/electron/package.json desktop/electron/tests/electron-smoke.spec.ts desktop/electron/tests/fixtures/stub-runtime.ts
git commit -m "test: add electron desktop smoke test"
```

---

### Task 9: Documentation and Upstream Merge Notes

Document usage, packaging assumptions, data locations, token guard, sandbox defaults, and upstream touch policy.

**Files:**
- Create: `docs/desktop-local-runtime.md`
- Modify: `README.md`

- [ ] **Step 1: Create desktop docs**

Create `docs/desktop-local-runtime.md`:

```markdown
# Desktop Local Runtime

The desktop shell runs DeerFlow's existing Gateway, Agent loop, and Sandbox on the customer's machine. The Electron main process owns only desktop lifecycle, local app-data paths, loopback ports, sidecar processes, and the browser-facing proxy.

## Data Paths

- macOS: `~/Library/Application Support/DeerFlow/`
- Windows: `%APPDATA%/DeerFlow/`

The desktop app writes `config.yaml`, `extensions_config.json`, `.env`, `desktop-token`, `better-auth-secret`, `.deer-flow/`, `logs/`, and `runtime/` under app-data.

## Developer Shell

```bash
cd desktop/electron
pnpm install
pnpm dev
```

Developer mode may rely on local `uv`, Python, Node.js, and pnpm.

## Packaged Runtime

Packaged mode stages existing DeerFlow artifacts into Electron resources:

```text
resources/
  backend/
    config.example.yaml
    extensions_config.example.json
  frontend/
  runtimes/
  desktop-server/
```

Build frontend standalone first:

```bash
cd frontend
NEXT_CONFIG_BUILD_OUTPUT=standalone SKIP_ENV_VALIDATION=1 pnpm build
```

Then stage desktop resources:

```bash
cd desktop/electron
pnpm build
export DEER_FLOW_DESKTOP_BUILD_PYTHON="$DESKTOP_PYTHON_RUNTIME_DIR/bin/python"
export DEER_FLOW_DESKTOP_NODE_RUNTIME_DIR="$DESKTOP_NODE_RUNTIME_DIR"
export DEER_FLOW_DESKTOP_PYTHON_RUNTIME_DIR="$DESKTOP_PYTHON_RUNTIME_DIR"
export DEER_FLOW_DESKTOP_PYTHON_SITE_PACKAGES_DIR="$DESKTOP_SITE_PACKAGES_DIR"
node dist/scripts/stage-python-deps.js ../../backend "$DEER_FLOW_DESKTOP_PYTHON_SITE_PACKAGES_DIR"
node dist/scripts/build-runtime.js ../.. resources
pnpm smoke resources
node dist/scripts/smoke-runtime.js ../.. --packaged --resources resources
```

`DESKTOP_NODE_RUNTIME_DIR`, `DESKTOP_PYTHON_RUNTIME_DIR`, and `DESKTOP_SITE_PACKAGES_DIR` are release-build inputs supplied by the operator or CI job for the target platform. On Windows build hosts, set `DEER_FLOW_DESKTOP_BUILD_PYTHON` to the Python runtime's `python.exe`. The runtime smoke uses a temporary app-data path containing spaces, initializes the first local admin through the desktop proxy, and verifies `/api/models` plus generated local-sandbox config.

## Security

Gateway desktop mode is enabled by `DEER_FLOW_DESKTOP=1`. In this mode Gateway requires `X-DeerFlow-Desktop-Token` for all API requests except `OPTIONS` and `/health`. The renderer never receives this token. The desktop proxy reads the token from app-data and injects it when forwarding requests to Gateway.

The existing setup/login/session cookies remain the user identity. The desktop token is only a same-machine transport guard.

## Sandbox

Desktop supports only `LocalSandboxProvider` in this MVP. The generated config keeps `allow_host_bash: false`. No other sandbox mode is supported by the desktop app.

## Upstream Touch List

- `backend/app/gateway/desktop_token_middleware.py`
- `backend/app/gateway/app.py`
- `backend/tests/test_desktop_token_middleware.py`
- `docs/desktop-local-runtime.md`
- `README.md`

All orchestration code lives under `desktop/electron/`.

## Smoke Matrix

| Platform | Sandbox | Status |
|---|---|---|
| Current platform packaged runtime smoke | LocalSandboxProvider | Required before completion |
| macOS installed path with spaces | LocalSandboxProvider | - [ ] |
| Windows installed path with spaces | LocalSandboxProvider | - [ ] |
```

- [ ] **Step 2: Link docs from README**

Modify `README.md` by adding one short line in the development or documentation section:

```markdown
- Desktop local runtime packaging notes: [docs/desktop-local-runtime.md](docs/desktop-local-runtime.md)
```

Place it under the existing `## Documentation` list, after `Backend Architecture`.

- [ ] **Step 3: Verify docs links**

Run: `test -f docs/desktop-local-runtime.md && rg -n "desktop-local-runtime" README.md docs/desktop-local-runtime.md`

Expected: both files are found.

- [ ] **Step 4: Commit**

```bash
git add docs/desktop-local-runtime.md README.md
git commit -m "docs: add desktop local runtime guide"
```

---

### Task 10: Final Verification

Run a high-signal verification set before handing off for platform smoke.

**Files:**
- No new files.

- [ ] **Step 1: Run backend focused verification**

Run:

```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_desktop_token_middleware.py -v
```

Expected: PASS.

- [ ] **Step 2: Run desktop unit verification**

Run:

```bash
cd desktop/electron
pnpm test
pnpm build
```

Expected: PASS.

- [ ] **Step 3: Run Electron smoke verification**

Run:

```bash
cd desktop/electron
pnpm test:electron
```

Expected: PASS.

- [ ] **Step 4: Run frontend standalone build verification**

Run:

```bash
cd frontend
NEXT_CONFIG_BUILD_OUTPUT=standalone SKIP_ENV_VALIDATION=1 pnpm build
test -f .next/standalone/server.js
```

Expected: Next build succeeds and `server.js` exists.

- [ ] **Step 5: Run packaged resource smoke**

Run:

```bash
cd desktop/electron
pnpm build
export DEER_FLOW_DESKTOP_BUILD_PYTHON="$DESKTOP_PYTHON_RUNTIME_DIR/bin/python"
export DEER_FLOW_DESKTOP_NODE_RUNTIME_DIR="$DESKTOP_NODE_RUNTIME_DIR"
export DEER_FLOW_DESKTOP_PYTHON_RUNTIME_DIR="$DESKTOP_PYTHON_RUNTIME_DIR"
export DEER_FLOW_DESKTOP_PYTHON_SITE_PACKAGES_DIR="$DESKTOP_SITE_PACKAGES_DIR"
node dist/scripts/stage-python-deps.js ../../backend "$DEER_FLOW_DESKTOP_PYTHON_SITE_PACKAGES_DIR"
node dist/scripts/build-runtime.js ../.. resources
pnpm smoke resources
node dist/scripts/smoke-runtime.js ../.. --packaged --resources resources
```

Expected: `Packaged runtime layout smoke passed` and `Desktop runtime smoke passed (packaged mode)`. `DESKTOP_NODE_RUNTIME_DIR`, `DESKTOP_PYTHON_RUNTIME_DIR`, and `DESKTOP_SITE_PACKAGES_DIR` are operator-supplied release-build inputs for the target platform. This is the required real Gateway + Next + desktop proxy product-path smoke for the current platform; separate macOS/Windows installer smoke rows remain manual release checks.

- [ ] **Step 6: Verify desktop remains local-sandbox only**

Run:

```bash
! rg -n "AioSandbox|Docker|docker|Apple Container|container_prefix" \
  desktop/electron/src desktop/electron/package.json desktop/electron/electron-builder.yml \
  docs/desktop-local-runtime.md
if git diff -- README.md | rg -n "AioSandbox|Docker|docker|Apple Container|container_prefix"; then
  echo "README desktop diff introduced non-local sandbox language"
  exit 1
fi
```

Expected: no matches in desktop product-path files, and the README diff does not introduce non-local desktop sandbox language. Negative fixtures under `desktop/electron/tests/` are allowed only when they assert the generated desktop config removes non-local sandbox settings. Existing non-desktop Docker documentation in README is out of scope for this desktop-only check.

- [ ] **Step 7: Inspect upstream touch list**

Run:

```bash
git diff --name-only main...HEAD
```

Expected: changes outside `desktop/electron/` are limited to:

```text
backend/app/gateway/app.py
backend/app/gateway/desktop_token_middleware.py
backend/tests/test_desktop_token_middleware.py
docs/desktop-local-runtime.md
README.md
```

- [ ] **Step 8: Commit any verification/doc adjustments**

If verification required doc or script adjustments, inspect the exact files first:

```bash
git status --short
git diff -- docs/desktop-local-runtime.md desktop/electron/src/scripts/build-runtime.ts desktop/electron/src/scripts/stage-python-deps.ts desktop/electron/src/scripts/smoke-packaged.ts desktop/electron/src/scripts/smoke-runtime.ts
```

Then stage only the verified files that changed and commit:

```bash
git commit -m "chore: finalize desktop runtime verification"
```

If no adjustments were needed, do not create an empty commit.

## Follow-Ups Not In This MVP Plan

- Native Keychain / Windows Credential Manager storage.
- Code signing and notarization.
- Auto-update channel.
- Cloud account/license/device binding.
- Remote worker protocol or cloud-synced thread state.
- PyInstaller or other Python binary freezing.
