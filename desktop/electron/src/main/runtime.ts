import { randomBytes } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";

import { ensureDesktopData, type DesktopDataPaths } from "./app-data.js";
import { buildGatewayEnv, buildNextEnv } from "./env.js";
import { allocateDesktopPorts } from "./ports.js";
import { resolveDesktopResources } from "./paths.js";
import { startSidecar, type SidecarProcess } from "./sidecar.js";
import { startDesktopProxy, type DesktopProxy } from "../proxy/desktop-proxy.js";

export type RuntimeCommand = {
  command: string;
  args: string[];
  cwd: string;
  env: NodeJS.ProcessEnv;
};

export type BuildGatewayCommandOptions = {
  packaged: boolean;
  backendDir: string;
  pythonBin: string;
  port: number;
  dataRoot: string;
};

export type BuildNextCommandOptions = {
  packaged: boolean;
  frontendDir: string;
  nodeBin: string;
  port: number;
  registerFetchPath?: string;
};

export type StartDesktopRuntimeOptions = {
  appDataRoot: string;
  packaged: boolean;
  appPath: string;
  resourcesPath: string;
};

export type DesktopRuntime = {
  proxyOrigin: string;
  stop: () => Promise<void>;
};

type RuntimePart = {
  stop: () => Promise<void>;
};

type RuntimeProxyPart = {
  close: () => Promise<void>;
};

const READY_TIMEOUT_MS = 60_000;
const CLEANUP_TIMEOUT_MS = 5_000;
const INTERNAL_NEXT_HEADER = "x-deerflow-desktop-internal-next";

export function buildGatewayCommand(options: BuildGatewayCommandOptions): RuntimeCommand {
  const pythonPath = options.packaged
    ? joinPaths([
        options.backendDir,
        join(options.backendDir, "packages", "harness"),
        join(options.backendDir, "site-packages"),
      ])
    : joinPaths([
        options.backendDir,
        join(options.backendDir, "packages", "harness"),
        process.env.PYTHONPATH,
      ]);

  if (options.packaged) {
    return {
      command: options.pythonBin,
      args: [
        "-m",
        "uvicorn",
        "app.gateway.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        String(options.port),
      ],
      cwd: options.dataRoot,
      env: {
        GATEWAY_PORT: String(options.port),
        PYTHONPATH: pythonPath,
      },
    };
  }

  return {
    command: "uv",
    args: [
      "--project",
      options.backendDir,
      "run",
      "uvicorn",
      "app.gateway.app:app",
      "--host",
      "127.0.0.1",
      "--port",
      String(options.port),
    ],
    cwd: options.dataRoot,
    env: {
      GATEWAY_PORT: String(options.port),
      PYTHONPATH: pythonPath,
    },
  };
}

export function buildNextCommand(options: BuildNextCommandOptions): RuntimeCommand {
  const importArgs = options.registerFetchPath ? ["--import", options.registerFetchPath] : [];
  const env: NodeJS.ProcessEnv = {
    PORT: String(options.port),
    HOSTNAME: "127.0.0.1",
  };

  if (options.registerFetchPath) {
    env.NODE_OPTIONS = joinNodeOptions([
      process.env.NODE_OPTIONS,
      `--import ${JSON.stringify(options.registerFetchPath)}`,
    ]);
  }

  if (options.packaged) {
    return {
      command: options.nodeBin,
      args: [...importArgs, join(options.frontendDir, ".next", "standalone", "server.js")],
      cwd: options.frontendDir,
      env,
    };
  }

  return {
    command: "pnpm",
    args: ["dev", "--hostname", "127.0.0.1", "--port", String(options.port)],
    cwd: options.frontendDir,
    env,
  };
}

export function classifyReadinessFailure(
  name: string,
  lastError: unknown,
  logTail?: string,
): string {
  const source = `${name}\n${errorMessage(lastError)}\n${logTail ?? ""}`;
  const lower = source.toLowerCase();

  if (
    lower.includes("modulenotfound") ||
    lower.includes("module not found") ||
    lower.includes("importerror") ||
    lower.includes("import error") ||
    lower.includes("failed to import")
  ) {
    return `${name} Gateway failed to import. Check packaged backend files and PYTHONPATH.`;
  }

  if (lower.includes("config") || lower.includes("validation") || lower.includes("yaml")) {
    return `${name} config invalid. Check config.yaml and extensions_config.json in desktop data.`;
  }

  if (
    lower.includes("no models") ||
    lower.includes("api key") ||
    lower.includes("missing model") ||
    lower.includes("missing_model") ||
    lower.includes("model key")
  ) {
    return `${name} model key missing. Add model provider credentials in the desktop .env or config.yaml.`;
  }

  if (lower.includes("frontend") || lower.includes("next")) {
    return `${name} frontend failed to start. Check the Next.js startup log.`;
  }

  if (
    lower.includes("localsandboxprovider") ||
    lower.includes("allow_host_bash") ||
    lower.includes("sandbox configuration")
  ) {
    return `${name} local sandbox configuration invalid. Check LocalSandboxProvider and allow_host_bash in config.yaml.`;
  }

  return `${name} failed to become ready: ${errorMessage(lastError)}`;
}

export async function waitForHttpReady(
  url: string,
  name: string,
  logPath: string,
  timeoutMs = READY_TIMEOUT_MS,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  let lastError: unknown = new Error("Timed out waiting for readiness");

  while (Date.now() < deadline) {
    try {
      const response = await fetchWithTimeout(url, 1_000);
      if (response.ok || response.status < 500) {
        return;
      }
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }

    await delay(250);
  }

  const logTail = await readLogTail(logPath);
  throw new Error(classifyReadinessFailure(name, lastError, logTail));
}

export async function startDesktopRuntime(
  options: StartDesktopRuntimeOptions,
): Promise<DesktopRuntime> {
  const resources = resolveDesktopResources(options);
  const paths = await ensureDesktopData({
    root: options.appDataRoot,
    exampleConfigPath: resources.repoRoot
      ? join(resources.repoRoot, "config.example.yaml")
      : join(resources.backendDir, "config.example.yaml"),
    exampleExtensionsConfigPath: resources.repoRoot
      ? join(resources.repoRoot, "extensions_config.example.json")
      : join(resources.backendDir, "extensions_config.example.json"),
  });
  const ports = await allocateDesktopPorts();
  const gatewayOrigin = `http://127.0.0.1:${ports.gatewayPort}`;
  const nextOrigin = `http://127.0.0.1:${ports.nextPort}`;
  const proxyOrigin = `http://127.0.0.1:${ports.proxyPort}`;
  const internalHeaderValue = randomBytes(32).toString("base64url");
  const registerFetchPath = join(resources.desktopServerDir, "next", "register-fetch.js");

  let gateway: SidecarProcess | null = null;
  let next: SidecarProcess | null = null;
  let proxy: DesktopProxy | null = null;

  try {
    gateway = await startGateway({
      packaged: options.packaged,
      backendDir: resources.backendDir,
      pythonBin: resources.pythonBin,
      port: ports.gatewayPort,
      paths,
      frontendOrigin: proxyOrigin,
    });
    await waitForHttpReady(`${gatewayOrigin}/health`, "gateway", join(paths.logsDir, "gateway.log"));

    proxy = await startDesktopProxy({
      host: "127.0.0.1",
      port: ports.proxyPort,
      gatewayOrigin,
      nextOrigin,
      tokenPath: paths.tokenPath,
      internalHeaderValue,
      logPath: join(paths.logsDir, "proxy.log"),
    });

    next = await startNext({
      packaged: options.packaged,
      frontendDir: resources.frontendDir,
      nodeBin: resources.nodeBin,
      port: ports.nextPort,
      paths,
      proxyOrigin,
      registerFetchPath,
      internalHeaderValue,
    });
    await waitForHttpReady(proxyOrigin, "frontend", join(paths.logsDir, "next.log"));

    return {
      proxyOrigin,
      stop: async () => {
        await stopRuntime(proxy, next, gateway);
      },
    };
  } catch (error) {
    throw await cleanupAfterStartupFailure(error, proxy, next, gateway);
  }
}

async function startGateway(options: {
  packaged: boolean;
  backendDir: string;
  pythonBin: string;
  port: number;
  paths: DesktopDataPaths;
  frontendOrigin: string;
}): Promise<SidecarProcess> {
  const baseEnv = await buildGatewayEnv(options.paths, { frontendOrigin: options.frontendOrigin });
  const command = buildGatewayCommand({
    packaged: options.packaged,
    backendDir: options.backendDir,
    pythonBin: options.pythonBin,
    port: options.port,
    dataRoot: options.paths.root,
  });

  return startSidecar({
    name: "gateway",
    command: command.command,
    args: command.args,
    cwd: command.cwd,
    env: { ...baseEnv, ...command.env },
    logPath: join(options.paths.logsDir, "gateway.log"),
  });
}

async function startNext(options: {
  packaged: boolean;
  frontendDir: string;
  nodeBin: string;
  port: number;
  paths: DesktopDataPaths;
  proxyOrigin: string;
  registerFetchPath: string;
  internalHeaderValue: string;
}): Promise<SidecarProcess> {
  const baseEnv = await buildNextEnv(options.paths, { proxyOrigin: options.proxyOrigin });
  const command = buildNextCommand({
    packaged: options.packaged,
    frontendDir: options.frontendDir,
    nodeBin: options.nodeBin,
    port: options.port,
    registerFetchPath: options.registerFetchPath,
  });

  return startSidecar({
    name: "next",
    command: command.command,
    args: command.args,
    cwd: command.cwd,
    env: {
      ...baseEnv,
      DEER_FLOW_INTERNAL_GATEWAY_HEADER_NAME: INTERNAL_NEXT_HEADER,
      DEER_FLOW_INTERNAL_GATEWAY_HEADER_VALUE: options.internalHeaderValue,
      ...command.env,
    },
    logPath: join(options.paths.logsDir, "next.log"),
  });
}

export async function stopRuntime(
  proxy: RuntimeProxyPart | null,
  next: RuntimePart | null,
  gateway: RuntimePart | null,
  cleanupTimeoutMs = CLEANUP_TIMEOUT_MS,
) {
  const errors: unknown[] = [];

  if (proxy) {
    try {
      await withTimeout(proxy.close(), cleanupTimeoutMs, "proxy close timed out");
    } catch (error) {
      errors.push(error);
    }
  }

  const sidecars: Array<RuntimePart | null> = [gateway, next];
  for (const sidecar of sidecars.reverse()) {
    if (sidecar) {
      try {
        await withTimeout(sidecar.stop(), cleanupTimeoutMs, "sidecar stop timed out");
      } catch (error) {
        errors.push(error);
      }
    }
  }

  if (errors.length > 0) {
    throw new AggregateError(errors, "runtime cleanup failed");
  }
}

export async function cleanupAfterStartupFailure(
  startupError: unknown,
  proxy: RuntimeProxyPart | null,
  next: RuntimePart | null,
  gateway: RuntimePart | null,
  cleanupTimeoutMs = CLEANUP_TIMEOUT_MS,
): Promise<unknown> {
  try {
    await stopRuntime(proxy, next, gateway, cleanupTimeoutMs);
    return startupError;
  } catch (cleanupError) {
    return new AggregateError([startupError, cleanupError], "desktop runtime startup failed during cleanup");
  }
}

function joinPaths(values: Array<string | undefined>): string {
  return values.filter((value): value is string => Boolean(value)).join(process.platform === "win32" ? ";" : ":");
}

function joinNodeOptions(values: Array<string | undefined>): string {
  return values.filter((value): value is string => Boolean(value)).join(" ");
}

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number, message: string): Promise<T> {
  let timeout: NodeJS.Timeout | undefined;
  const timeoutPromise = new Promise<never>((_resolve, reject) => {
    timeout = setTimeout(() => reject(new Error(message)), timeoutMs);
    timeout.unref();
  });

  try {
    return await Promise.race([promise, timeoutPromise]);
  } finally {
    if (timeout) {
      clearTimeout(timeout);
    }
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function readLogTail(logPath: string): Promise<string> {
  try {
    const content = await readFile(logPath, "utf8");
    return content.slice(-8_192);
  } catch {
    return "";
  }
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms).unref();
  });
}

async function fetchWithTimeout(url: string, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  timeout.unref();

  try {
    return await fetch(url, {
      cache: "no-store",
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeout);
  }
}
