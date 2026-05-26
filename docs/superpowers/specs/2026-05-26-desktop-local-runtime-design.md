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
| Sandbox | Abstract `SandboxProvider`; existing `LocalSandboxProvider` and `AioSandboxProvider` | The desktop app can choose local or container sandbox via config, without adding a new sandbox protocol. |
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

Use **Approach A: Electron thin shell with local DeerFlow sidecars** for phase one.

New code should live under `desktop/electron/` and related documentation/scripts. Existing DeerFlow source should only receive small, explicit compatibility hooks when unavoidable.

### 4.1 Process Model

Desktop startup:

1. Electron main process resolves the OS app-data directory:
   - macOS: `~/Library/Application Support/DeerFlow/`
   - Windows: `%APPDATA%/DeerFlow/`
2. It ensures these local files/directories exist:
   - `config.yaml`
   - `extensions_config.json`
   - `.env`
   - `.deer-flow/`
   - `logs/`
3. It allocates loopback ports for Gateway and frontend.
4. It starts Gateway as a local sidecar:
   - Host: `127.0.0.1`
   - Dynamic port chosen by Electron
   - App env:
     - `DEER_FLOW_HOME=<app-data>/.deer-flow`
     - `DEER_FLOW_CONFIG_PATH=<app-data>/config.yaml`
     - `DEER_FLOW_EXTENSIONS_CONFIG_PATH=<app-data>/extensions_config.json`
     - `GATEWAY_CORS_ORIGINS=<desktop frontend origin>`
     - `DEER_FLOW_DESKTOP=1`
5. It starts or serves the packaged frontend:
   - Development: point to `next dev` or an externally running dev URL.
   - Packaged build: run Next standalone server or an equivalent packaged web server.
6. It opens a BrowserWindow to the local frontend URL.
7. On app quit, it terminates child processes and best-effort cleans up running sandbox containers that belong to the desktop app.

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

### 4.4 Sandbox Strategy

Use existing sandbox providers:

| Mode | Use case | Default stance |
|---|---|---|
| `LocalSandboxProvider` | Simple single-user local workflows | Supported, but keep `allow_host_bash: false` by default. |
| `AioSandboxProvider` with Docker/Apple Container | Better isolation for code execution | Supported for users who install Docker Desktop or Apple Container. |
| K8s/provisioner mode | Enterprise managed sandbox pools | Out of scope for desktop MVP unless already configured by a customer. |

The desktop shell should not bypass `SandboxProvider` or call host commands directly on behalf of agent tools.

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
2. A desktop-local auth middleware switch, guarded by `DEER_FLOW_DESKTOP=1`.
3. Frontend config fallback for desktop-provided local URLs.
4. Documentation of desktop mode in README/install docs.

Any core hook should be small, feature-flagged, covered by tests, and explain why the desktop shell cannot handle it externally.

## 6. Security Design

Desktop mode creates a local HTTP surface. It must not assume that "localhost" is automatically private.

Required phase-one controls:

- Gateway binds to `127.0.0.1`, not `0.0.0.0`.
- Electron generates a per-install or per-session local secret.
- Requests from the packaged frontend to Gateway carry that secret through a cookie or header.
- Gateway verifies the secret only when `DEER_FLOW_DESKTOP=1`.
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
- Enterprise policy file for disabling host-local sandbox modes.

## 7. Configuration and Data

Desktop data should be isolated from the source checkout:

```text
<app-data>/DeerFlow/
  config.yaml
  extensions_config.json
  .env
  .deer-flow/
    memory.json
    users/
    threads/
  logs/
  runtime/
```

Config creation:

- First run copies from `config.example.yaml` and `extensions_config.example.json` when present.
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

Two packaging options should be evaluated during implementation planning:

| Option | Description | Trade-off |
|---|---|---|
| Bundled Python + `uv` environment | Ship Python runtime plus synced backend environment | Closer to source layout, easier upstream merging, larger install and slower first launch. |
| PyInstaller Gateway binary | Build Gateway into a platform-specific executable | Simpler runtime startup, but higher risk from dynamic imports, LangChain providers, MCP, and config-driven class loading. |

Recommendation for first implementation plan: start with **bundled runtime/source layout** because it preserves DeerFlow's dynamic provider model and avoids fighting PyInstaller too early.

The sandbox container image should not be bundled in the installer. The desktop app should detect Docker/Apple Container and offer a clear setup/pull flow.

## 9. Testing Strategy

Core regression tests:

- Backend tests continue to run unchanged.
- Add focused tests only for any desktop-mode Gateway hook:
  - local secret accepted/rejected
  - CORS/CSRF behavior in desktop mode
  - Gateway host/port config if changed

Desktop integration tests:

- Main process chooses ports and starts/stops sidecars.
- App-data paths are created and passed via env.
- Frontend can load and fetch `/api/models`.
- LangGraph stream endpoint can start a mock/minimal run when config is valid.
- Quit cleans child processes.

Manual cross-platform smoke matrix:

| Platform | Sandbox | Expected smoke |
|---|---|---|
| macOS | LocalSandboxProvider | App launches, model list loads, local thread data created under app-data. |
| macOS | AioSandboxProvider + Apple Container or Docker | Sandbox health passes and generated artifacts appear under app-data. |
| Windows | LocalSandboxProvider | App launches without Git Bash dependency in packaged mode. |
| Windows | AioSandboxProvider + Docker Desktop | Sandbox starts and file mounts use Windows-safe host paths. |

## 10. Rollout Plan

1. **Phase 0: Design and implementation plan**
   - Finalize this design.
   - Write a task-by-task implementation plan.
2. **Phase 1: Developer desktop shell**
   - Electron app starts existing dev Gateway/frontend.
   - No production packaging yet.
3. **Phase 2: Packaged local runtime**
   - Ship Gateway/frontend as sidecars.
   - Desktop app-data config and logs.
   - macOS/Windows smoke tests.
4. **Phase 3: Hardened desktop mode**
   - Local secret verification.
   - Native credential storage.
   - update/signing/notarization.
5. **Phase 4: Optional cloud coordination**
   - Config sync, license/device management, optional thread sync.

## 11. Open Questions

- Should desktop MVP require Docker/AIO sandbox for customer deployments, or allow local sandbox by default with clear warnings?
- Should packaged mode use Next standalone server, static export where possible, or Electron's custom protocol plus API proxy?
- Should cloud account login be required before local execution, or only before sync/licensed features?
- What is the target installer/update system: Electron Forge, electron-builder, or a company-standard pipeline?
- Are enterprise customers expected to run behind corporate proxies that require proxy config for model APIs, MCP, and container image pulls?

## 12. Decision

Proceed with an Electron thin shell around the existing local Gateway and frontend, keeping Agent loop and Sandbox execution inside the current DeerFlow runtime. Prioritize new peripheral files over edits to upstream-tracked core modules.
