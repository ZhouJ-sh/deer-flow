# Desktop Local Runtime Shell Design

**Date**: 2026-05-26
**Status**: Design approved, implementation plan pending
**Primary constraint**: Minimize changes to existing upstream-tracked source so future upstream merges stay cheap.

---

## 1. Goal

Add a Windows and macOS desktop shell that runs DeerFlow's Agent loop and Sandbox on the customer's machine, reducing server-side CPU, memory, and sandbox load while preserving the existing Gateway, frontend, LangGraph-compatible API, and harness code as much as possible.

The first phase targets a local execution desktop app, not a full rewrite into native desktop architecture. Cloud services may remain responsible for lightweight coordination such as account state, license checks, configuration distribution, and optional future sync.

## 2. Current Architecture Findings

DeerFlow 2.0 already has a favorable boundary for desktop migration:

| Area | Current state | Desktop implication |
|---|---|---|
| Agent runtime | Embedded in Gateway via `RunManager`, `run_agent()`, and `StreamBridge` under `backend/packages/harness/deerflow/runtime/` | A desktop app can start the existing Gateway locally instead of splitting or rewriting the Agent loop. |
| Public agent API | Frontend talks to `/api/langgraph/*`; nginx or Next rewrites that to Gateway's native `/api/*` routers | The desktop shell can preserve the same browser/API contract. |
| Sandbox | Abstract `SandboxProvider`; existing `LocalSandboxProvider` | The desktop MVP uses local sandbox only, avoiding extra runtime packaging and cleanup concerns. |
| User data paths | `DEER_FLOW_HOME` and `Paths` centralize runtime state and per-thread sandbox directories | Desktop data can live in the OS app-data directory without writing back to the repo checkout. |
| Frontend | Next.js app uses current origin plus rewrites/env vars for Gateway routing | Desktop can load a local Next server or packaged frontend while keeping frontend API calls mostly unchanged. |
| Development scripts | `make dev` starts Gateway + Frontend + nginx; Windows local flow already expects Git Bash for scripts | Desktop startup should not depend on nginx or shell scripts in production packaging. |

## 3. Approaches Considered

### A. Electron thin shell with local DeerFlow sidecars

Add a new `desktop/electron/` package. Electron main process owns desktop lifecycle, launches the existing Gateway and frontend server on loopback ports, injects desktop-specific environment variables, and loads the local app URL.

Pros:
- Lowest impact on existing `backend/`, `frontend/`, and `harness` code.
- Uses the existing web UI and API contracts.
- Mature process-management and packaging ecosystem.
- Easier first milestone on both macOS and Windows.

Cons:
- Larger app bundle.
- Needs careful process cleanup and update strategy.

### B. Tauri thin shell with local DeerFlow sidecars

Add a Tauri app that launches the same Gateway/frontend sidecars.

Pros:
- Smaller shell and native platform feel.
- Tauri v2 supports sidecar binaries.

Cons:
- Adds Rust/Tauri configuration and permission surface.
- More packaging variance for Python/Node sidecars.
- Not clearly lower-risk for a first migration, given the priority is upstream mergeability and fast delivery.

### C. Cloud UI with local worker agent

Keep the web app hosted remotely. Install a local worker that receives jobs from the cloud and executes Agent loop/Sandbox locally.

Pros:
- Server load reduction can be strong.
- Centralized web updates remain simple.

Cons:
- Requires new device pairing, authenticated worker protocol, task routing, reconnect behavior, and state synchronization.
- Much larger change to core runtime/API boundaries.
- Higher risk of upstream conflicts.

## 4. Recommended Architecture

Use **Approach A: Electron thin shell with local DeerFlow sidecars**.

New code should live under `desktop/electron/` and related documentation/scripts. Existing DeerFlow source should only receive small, explicit compatibility hooks when unavoidable.

There are two explicit delivery gates:

| Phase | Audience | Security bar |
|---|---|---|
| Developer shell | Contributors only | May rely on local dev servers and existing auth behavior. Not shippable to customers. |
| Packaged customer runtime | Customers | Must include the desktop local access token, existing user auth/setup, loopback-only Gateway binding, concrete frontend serving mode, and packaged runtime artifacts. |

### 4.1 Process Model

Desktop startup:

1. Electron main process resolves the OS app-data directory:
   - macOS: `~/Library/Application Support/DeerFlow/`
   - Windows: `%APPDATA%/DeerFlow/`
   - This document calls that resolved directory `<desktop-data>`.
2. It ensures these local files/directories exist:
   - `config.yaml`
   - `extensions_config.json`
   - `.env`
   - `.deer-flow/`
   - `logs/`
3. It allocates loopback ports for Gateway, Next standalone, and the desktop frontend proxy.
4. It starts Gateway as a local sidecar:
   - Host: `127.0.0.1`
   - Dynamic port chosen by Electron
   - Working directory: `<desktop-data>/`, not the bundled backend resource directory
   - App env:
     - `DEER_FLOW_HOME=<desktop-data>/.deer-flow`
     - `DEER_FLOW_CONFIG_PATH=<desktop-data>/config.yaml`
     - `DEER_FLOW_EXTENSIONS_CONFIG_PATH=<desktop-data>/extensions_config.json`
     - `DEER_FLOW_PROJECT_ROOT=<desktop-data>/`
     - `DEER_FLOW_DESKTOP_TOKEN_FILE=<desktop-data>/desktop-token`
     - `GATEWAY_CORS_ORIGINS=<desktop frontend origin>`
     - `DEER_FLOW_DESKTOP=1`
5. It starts or serves the frontend:
   - Developer shell: point to `next dev` or an externally running dev URL.
   - Packaged customer runtime: run the packaged Next standalone server on an internal loopback port, then run a desktop HTTP proxy as the browser-facing origin.
6. It opens a BrowserWindow to the local frontend URL.
7. On app quit, it terminates child processes. No extra sandbox cleanup is needed because desktop MVP only supports `LocalSandboxProvider`.

### 4.2 Runtime Ownership

The Gateway remains the owner of:

- LangGraph-compatible runs and threads API
- `RunManager`
- `run_agent()`
- `StreamBridge`
- checkpointer/store/event-store initialization
- model factory
- MCP and skills APIs
- upload/artifact APIs
- sandbox lifecycle

Electron owns only:

- Process lifecycle
- OS app-data paths
- Port discovery
- desktop startup diagnostics
- update/install UX
- optional native credential storage in a later phase

### 4.3 API Flow

The frontend keeps using the current API shape:

```text
BrowserWindow
  -> local frontend origin
  -> /api/langgraph/* and /api/*
  -> local Gateway on 127.0.0.1:<gateway-port>
  -> existing Gateway routers and runtime
```

No new Agent API should be introduced in phase one.

### 4.4 Frontend Serving Decision

Packaged customer runtime uses **Next standalone**, not static export or a custom Electron protocol.

Rationale:

- The current frontend uses SSR auth checks and Next rewrites.
- Static export would require broader frontend changes.
- Next standalone keeps the existing `/api/langgraph/*` and `/api/*` same-origin shape.

Packaged startup requirements:

- Build frontend with `NEXT_CONFIG_BUILD_OUTPUT=standalone`.
- Bundle a Node runtime compatible with the built Next server.
- Start `.next/standalone/server.js` on `127.0.0.1:<next-port>` as an internal server.
- Start a desktop-only HTTP proxy on `127.0.0.1:<frontend-port>` as the BrowserWindow origin.
- The desktop proxy forwards page/static requests to the internal Next server and forwards all `/api/*` and `/api/langgraph/*` requests to Gateway.
- Set `DEER_FLOW_INTERNAL_GATEWAY_BASE_URL=http://127.0.0.1:<frontend-port>/_desktop-gateway` before starting the Next server, so SSR auth/setup calls go through the same desktop proxy path that injects the desktop token.
- Persist or generate `BETTER_AUTH_SECRET` under the desktop app-data directory.
- Do not set `NEXT_PUBLIC_LANGGRAPH_BASE_URL` or `NEXT_PUBLIC_BACKEND_BASE_URL` for the packaged MVP; client code should keep using the current same-origin defaults.
- Configure Gateway CORS/CSRF origins to the desktop proxy origin.
- The desktop proxy, not `next.config.js` rewrites alone, injects the desktop token into proxied Gateway requests.
- The proxy exposes `/_desktop-gateway/*` for local Next SSR server-side fetches only. Browser navigations or XHR/fetch requests to that prefix are rejected using request metadata such as `Origin`, `Sec-Fetch-*`, or an internal header set only by the Next server wrapper.

### 4.5 Sandbox Strategy

Desktop supports **only `LocalSandboxProvider`**.

| Mode | Use case | Default stance |
|---|---|---|
| `LocalSandboxProvider` | Single-user customer desktop workflows | Supported; keep `allow_host_bash: false` by default. |

The desktop shell should not bypass `SandboxProvider` or call host commands directly on behalf of agent tools. No container or remote sandbox runtime is bundled or orchestrated by the desktop app.

## 5. Minimal Core Changes

To preserve upstream mergeability, phase one should aim for:

### Prefer adding

- `desktop/electron/package.json`
- `desktop/electron/src/main/*`
- `desktop/electron/src/preload/*`
- `desktop/electron/resources/*`
- `desktop/electron/scripts/*`
- desktop docs under `docs/` or frontend docs

### Avoid changing

- `backend/packages/harness/deerflow/runtime/*`
- `backend/packages/harness/deerflow/agents/*`
- `backend/packages/harness/deerflow/sandbox/*`
- frontend chat/runtime hooks
- existing deployment scripts beyond optional references

### Acceptable small hooks

Only if needed:

1. Gateway host/port/env defaults that allow binding to `127.0.0.1:<dynamic-port>`.
2. A desktop token guard middleware switch, guarded by `DEER_FLOW_DESKTOP=1`.
3. Frontend config fallback for desktop-provided local URLs.
4. Documentation of desktop mode in README/install docs.

Any core hook should be small, feature-flagged, covered by tests, and explain why the desktop shell cannot handle it externally.

### Upstream Touch Policy

Every change outside `desktop/electron/` must be tracked in an "upstream touch list" in the implementation PR:

| Touch type | Rule |
|---|---|
| Gateway hook | Guard behind `DEER_FLOW_DESKTOP=1` or a narrower `DEER_FLOW_DESKTOP_*` env var. |
| Frontend hook | Prefer one small desktop compatibility helper over edits spread through chat/runtime components. |
| Config docs | Keep examples additive; do not rewrite existing deployment docs. |
| Tests | Add a focused regression test for each hook. |
| Ownership | Record file, reason, and removal criteria in the PR or implementation plan. |

## 6. Security Design

Desktop mode creates a local HTTP surface. It must not assume that "localhost" is automatically private.

Required controls before any packaged customer runtime:

- Gateway binds to `127.0.0.1`, not `0.0.0.0`.
- Electron generates a per-install desktop local access token and stores it in the desktop app-data directory with user-only filesystem permissions where the OS supports them.
- Electron passes `DEER_FLOW_DESKTOP_TOKEN_FILE=<desktop-data>/desktop-token` to both Gateway and the desktop HTTP proxy. Gateway reads the expected token from that file; the proxy reads the same file to inject the request header.
- Startup fails closed if the desktop token file is missing, empty, unreadable, world-readable on platforms where permissions can be checked, or different between Gateway/proxy reads. Token rotation is an explicit logout/reset operation that restarts both sidecars after writing the new file.
- The packaged Next frontend proxy is the only browser-facing origin. A desktop-only proxy forwards all `/api/*` and `/api/langgraph/*` requests to Gateway.
- The desktop proxy, not renderer JavaScript, attaches `X-DeerFlow-Desktop-Token` when proxying to Gateway.
- Gateway verifies `X-DeerFlow-Desktop-Token` only when `DEER_FLOW_DESKTOP=1`; API requests missing or failing the token are rejected before normal CSRF/auth processing.
- The only Gateway paths exempt from the desktop token are `OPTIONS` preflight and `/health`. Setup, login, register, logout, and initialize endpoints still require the desktop token because they create or mutate local auth state.
- Renderer JavaScript never receives the desktop token, and the token is never stored in localStorage.
- CORS and CSRF settings allow only the desktop frontend origin.
- The BrowserWindow uses conservative settings:
  - no Node integration in renderer
  - context isolation enabled
  - preload exposes only a tiny desktop status/config API if needed
  - external links open in the system browser
- `LocalSandboxProvider` keeps host bash disabled unless the user explicitly opts into trusted local execution.
- Logs should redact secrets and API keys.

Deferred security enhancements:

- Store API keys in macOS Keychain / Windows Credential Manager.
- Code signing and notarization.
- Device attestation or cloud license binding.

### Desktop Auth Decision

Desktop mode keeps DeerFlow's existing user auth/setup flow. The desktop local access token is only a same-machine transport guard; it is not a user identity and must not replace `access_token` session cookies.

Request order in packaged customer runtime:

1. CORS handles browser origin preflight for the local frontend origin.
2. Desktop token guard rejects Gateway API requests without the correct `X-DeerFlow-Desktop-Token` when `DEER_FLOW_DESKTOP=1`; only `OPTIONS` and `/health` are exempt.
3. Existing `CSRFMiddleware` validates state-changing browser requests with the normal `csrf_token` double-submit flow.
4. Existing `AuthMiddleware` validates the normal `access_token` cookie or internal auth token and stamps `request.state.user`.
5. Existing route-level authorization and user isolation continue to operate unchanged.

First-run setup remains the existing `/setup` / `/api/v1/auth/initialize` flow. The first local admin user becomes the owner for desktop thread state. No synthetic desktop user should be introduced in the MVP because that would fork user isolation behavior from the existing Gateway model.

## 7. Configuration and Data

Desktop data should be isolated from the source checkout:

```text
<desktop-data>/
  config.yaml
  extensions_config.json
  .env
  desktop-token
  better-auth-secret
  .deer-flow/
    .jwt_secret
    data/
      deerflow.db
    memory.json
    users/
      <desktop-admin-user-id>/
        memory.json
        threads/
          <thread-id>/
            user-data/
              workspace/
              uploads/
              outputs/
            acp-workspace/
    threads/              # legacy/no-auth fallback only
  logs/
  runtime/
```

Config creation:

- First run copies from `config.example.yaml` and `extensions_config.example.json` when present.
- Electron parses `<desktop-data>/.env` itself and injects those values into the Gateway and frontend server environments. Desktop packaged mode must not rely on `load_dotenv()` discovering the app-data `.env` from whatever CWD the child process happens to use.
- Electron persists `BETTER_AUTH_SECRET` for the frontend and preserves Gateway's `AUTH_JWT_SECRET` either by writing it to `<desktop-data>/.env` or by allowing the existing Gateway fallback to persist `<desktop-data>/.deer-flow/.jwt_secret` via `DEER_FLOW_HOME`.
- Gateway packaged-mode CWD is `<desktop-data>/`; desktop config must also write absolute storage paths so future CWD changes cannot leak state into the bundled resources or source checkout.
- Desktop-generated `config.yaml` must set:
  - `database.backend: sqlite`
  - `database.sqlite_dir: <desktop-data>/.deer-flow/data`
  - `run_events.backend: db`
  - `checkpointer: null` or omit the legacy `checkpointer` section so unified `database` owns LangGraph state too
  - `sandbox.use: deerflow.sandbox.local:LocalSandboxProvider`
  - `sandbox.allow_host_bash: false`
- A desktop setup screen or launch diagnostic should guide missing model API keys.
- Existing `config_version` and `make config-upgrade` logic should be reused where practical, but desktop startup should not require Make.

## 8. Packaging Strategy

### Development

- Electron runs from `desktop/electron`.
- It can point to existing dev servers:
  - Gateway: `cd backend && uv run uvicorn app.gateway.app:app ...`
  - Frontend: `cd frontend && pnpm dev`
- This mode is for contributors only and may rely on local Node/Python/uv.

### Packaged MVP

The packaged MVP should use a prebuilt per-platform runtime/source layout, not first-launch dependency installation and not PyInstaller. Runtime artifacts must be relocatable across installation paths, including paths with spaces.

Artifact layout:

```text
resources/
  backend/
    config.example.yaml
    extensions_config.example.json
    app/
    packages/
    pyproject.toml
    uv.lock
    site-packages/          # wheel-installed pure/Python extension deps for the target platform/arch
  frontend/
    .next/standalone/
      .next/static/
      public/
  runtimes/
    node/
    python/
  desktop-server/
    proxy.js               # Browser-facing HTTP proxy + API header injection
```

Packaging rules:

- Build separate artifacts for macOS arm64, macOS x64, Windows x64, and Windows arm64 only if product requires it.
- Resolve and install Python dependencies at build time from the checked-in lockfile and any required extras; do not run `uv sync` on the customer's machine for the MVP.
- Do not copy a regular `.venv` into the app bundle as the launch mechanism. Virtual environments often bake absolute interpreter paths. Instead, launch the bundled Python interpreter with explicit `PYTHONPATH` entries for `resources/backend`, `resources/backend/packages/harness`, and the packaged dependency directory, or use a relocatable virtualenv format that is proven by install-path smoke tests.
- Gateway launch command:
  - macOS: `<resources>/runtimes/python/bin/python -m uvicorn app.gateway.app:app --host 127.0.0.1 --port <gateway-port>`
  - Windows: `<resources>\\runtimes\\python\\python.exe -m uvicorn app.gateway.app:app --host 127.0.0.1 --port <gateway-port>`
  - `cwd=<desktop-data>/`
  - `PYTHONPATH=<resources>/backend;<resources>/backend/packages/harness;<resources>/backend/site-packages` using the platform path separator
- Include certificates needed by Python HTTP clients or ensure the bundled Python uses the OS trust store consistently.
- Preserve DeerFlow source layout and dynamic imports so LangChain providers, MCP configuration, and `resolve_class()` continue to work.
- Do not use PyInstaller in the MVP. It remains a later optimization after dynamic-provider compatibility is proven.
- Startup diagnostics must distinguish "Gateway failed to import", "config invalid", "model key missing", "frontend failed to start", and "local sandbox configuration invalid".

No separate sandbox runtime is bundled in the installer for the MVP; desktop execution uses `LocalSandboxProvider`.

## 9. Testing Strategy

Core regression tests:

- Backend tests continue to run unchanged.
- Add focused tests only for any desktop-mode Gateway hook:
  - desktop token accepted/rejected
  - setup/login/register endpoints rejected without the desktop token in desktop mode
  - CORS/CSRF behavior in desktop mode
  - Gateway host/port config if changed

Desktop integration tests:

- Main process chooses ports and starts/stops sidecars.
- App-data paths are created and passed via env.
- Frontend can load and fetch `/api/models`.
- LangGraph stream endpoint can start a mock/minimal run when config is valid.
- Quit cleans child processes.
- Desktop token is attached by the desktop HTTP proxy and rejected when missing or incorrect.
- Existing setup/login flow still creates an admin user and stamps user-scoped thread data.
- `deerflow.db`, users, runs, run events, feedback, memory, and thread sandbox data all land under `<desktop-data>/.deer-flow`, not under the bundled resource directory or repo checkout.
- App-data `.env`, `BETTER_AUTH_SECRET`, and Gateway JWT secret survive app restart and app update.
- Port collision after allocation is handled by retrying or showing a clear launch error.
- Stale Gateway/frontend child processes from a crashed prior launch are detected and handled without killing unrelated processes.
- Corrupted `config.yaml` produces a recoverable diagnostic instead of a blank window.
- Missing model API keys send the user to setup/config diagnostics, not a generic sidecar failure.
- Windows paths with spaces work for app-data, backend resources, frontend resources, and sandbox mounts.
- Packaged runtime launches successfully from installed paths containing spaces on macOS and Windows.
- Desktop app update migrates or preserves `config.yaml`, `extensions_config.json`, `desktop-token`, `better-auth-secret`, and `.deer-flow`.

Manual cross-platform smoke matrix:

| Platform | Sandbox | Expected smoke |
|---|---|---|
| macOS | LocalSandboxProvider | App launches, model list loads, local thread data created under app-data. |
| Windows | LocalSandboxProvider | App launches without Git Bash dependency in packaged mode. |

Installer smoke:

- macOS unsigned developer build launches from the app bundle.
- macOS signed/notarized build launches without quarantine surprises once signing is enabled.
- Windows unpacked developer build launches.
- Windows installer build launches, writes app-data to `%APPDATA%`, and uninstalls without deleting user data unless explicitly requested.

## 10. Rollout Plan

1. **Phase 0: Design and implementation plan**
   - Finalize this design.
   - Write a task-by-task implementation plan.
2. **Phase 1: Developer desktop shell**
   - Electron app starts existing dev Gateway/frontend.
   - Contributor-only; may rely on local Node/Python/uv.
   - No customer packaging and no server-load-reduction claims yet.
3. **Phase 2: Packaged local runtime**
   - Ship Gateway/frontend as sidecars.
   - Desktop app-data config and logs.
   - Desktop local token guard.
   - Existing setup/login flow verified in desktop mode.
   - macOS/Windows smoke tests.
4. **Phase 3: Hardened desktop mode**
   - Native credential storage.
   - update/signing/notarization.
5. **Phase 4: Optional cloud coordination**
   - Config sync, license/device management, optional thread sync.

## 11. Open Questions

- Should cloud account login be required before local execution, or only before sync/licensed features?
- What is the target installer/update system: Electron Forge, electron-builder, or a company-standard pipeline?
- Are enterprise customers expected to run behind corporate proxies that require proxy config for model APIs and MCP?

## 12. Decision

Proceed with an Electron thin shell around the existing local Gateway and frontend, keeping Agent loop and Sandbox execution inside the current DeerFlow runtime. Prioritize new peripheral files over edits to upstream-tracked core modules.
