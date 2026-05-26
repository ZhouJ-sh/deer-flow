# Desktop Local Runtime

The desktop shell runs DeerFlow's existing Gateway, agent loop, and sandbox on the customer's machine. Electron owns desktop lifecycle, app-data paths, loopback ports, sidecar processes, packaged resources, and the browser-facing proxy. The backend runtime, `RunManager`, `run_agent()`, `StreamBridge`, and sandbox provider contract stay in the existing DeerFlow code.

## Data Paths

- macOS: `~/Library/Application Support/DeerFlow/`
- Windows: `%APPDATA%/DeerFlow/`

These are the packaged app-data locations for product name `DeerFlow`; developer builds still use Electron's current `app.getPath("userData")`.

The desktop app writes `config.yaml`, `extensions_config.json`, `.env`, `desktop-token`, `install-id`, `.deer-flow/`, `logs/`, and `runtime/` under app-data. The Electron env helper also creates `.deer-flow/better-auth-secret` for local session signing before starting Next.

## Developer Shell

```bash
cd desktop/electron
pnpm install
pnpm dev
```

Developer mode may rely on local `uv`, Python, Node.js, and pnpm. The Electron smoke test uses a stub runtime and does not require model API keys:

```bash
cd desktop/electron
pnpm build
pnpm test:electron
```

## Packaged Runtime

Packaged mode stages existing DeerFlow artifacts into Electron resources:

```text
resources/
  backend/
    app/
    packages/
    site-packages/
    pyproject.toml
    uv.lock
    config.example.yaml
    extensions_config.example.json
  frontend/
    .next/standalone/
      server.js
      .next/static/
      public/
  runtimes/
    node/
    python/
  desktop-server/
  skills/
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

`DESKTOP_NODE_RUNTIME_DIR`, `DESKTOP_PYTHON_RUNTIME_DIR`, and `DESKTOP_SITE_PACKAGES_DIR` are release-build inputs supplied by the operator or CI job for the target platform. On Windows build hosts, set `DEER_FLOW_DESKTOP_BUILD_PYTHON` to the Python runtime's `python.exe`. Dependency installation and runtime folder creation happen on release build machines, not customer machines.

## Security

Gateway desktop mode is enabled by `DEER_FLOW_DESKTOP=1`. In this mode Gateway requires `X-DeerFlow-Desktop-Token` for API requests except `OPTIONS` and `/health`. The renderer never receives this token. The desktop proxy reads it from app-data and injects it when forwarding requests to Gateway.

The existing setup, login, and session cookies remain the user identity. The desktop token is only a same-machine transport guard against direct loopback Gateway calls. It is not a boundary against code already running as the same OS user or anything that can read the app-data directory. Secret files are written with user-only permissions on POSIX and best-effort permissions on Windows.

## Sandbox

Desktop supports only `LocalSandboxProvider` in this MVP. This is host-backed local execution state, not container isolation. Generated desktop config sets:

```yaml
sandbox:
  use: deerflow.sandbox.local:LocalSandboxProvider
  allow_host_bash: false
```

With this provider, file and tool data map to local per-thread directories. `allow_host_bash: false` disables the bash tool by default because host bash is not an isolation boundary. The desktop shell does not bundle or orchestrate container sandboxes. It also does not bypass `SandboxProvider` or call host commands directly on behalf of agent tools.

## Upstream Touch List

- `backend/app/gateway/desktop_token_middleware.py`
- `backend/app/gateway/app.py`
- `backend/packages/harness/deerflow/config/app_config.py`
- `backend/tests/test_app_config_reload.py`
- `backend/tests/test_desktop_token_middleware.py`
- `docs/desktop-local-runtime.md`
- `README.md`

All desktop orchestration code lives under `desktop/electron/`. The backend touch points are feature-flagged by `DEER_FLOW_DESKTOP=1`; that flag enables the loopback token guard and rejects non-`LocalSandboxProvider` config during desktop config loads and reloads.

## Smoke Matrix

| Platform | Sandbox | Status |
|---|---|---|
| Current platform Electron stub smoke | LocalSandboxProvider | Passed with `pnpm test:electron` |
| Current platform packaged runtime smoke | LocalSandboxProvider | Requires operator-supplied runtime folders |
| macOS installed path with spaces | LocalSandboxProvider | - [ ] |
| Windows installed path with spaces | LocalSandboxProvider | - [ ] |
