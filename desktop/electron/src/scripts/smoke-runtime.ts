import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { parse } from "yaml";

import { startDesktopRuntime, type DesktopRuntime } from "../main/runtime.js";

type CookieJar = Map<string, string>;
export type SmokeRuntimeArgs = {
  appPath: string;
  resourcesPath: string;
  packaged: boolean;
};

export async function smokeRuntime(options: {
  appPath: string;
  resourcesPath: string;
  packaged: boolean;
}): Promise<void> {
  const appDataRoot = await mkdtemp(join(tmpdir(), "deer-flow desktop smoke "));
  let runtime: DesktopRuntime | null = null;

  try {
    runtime = await startDesktopRuntime({
      appDataRoot,
      packaged: options.packaged,
      appPath: options.appPath,
      resourcesPath: options.resourcesPath,
    });

    const cookies: CookieJar = new Map();
    await requestJson(`${runtime.proxyOrigin}/api/v1/auth/setup-status`, { cookies });
    await requestJson(`${runtime.proxyOrigin}/api/v1/auth/initialize`, {
      method: "POST",
      cookies,
      body: {
        email: `desktop-smoke-${Date.now()}@example.invalid`,
        password: "DesktopSmokePass123",
        confirm_password: "DesktopSmokePass123",
      },
      tolerateStatuses: [201, 409],
    });
    await requestJson(`${runtime.proxyOrigin}/api/models`, { cookies });

    await assertLocalSandboxConfig(join(appDataRoot, "config.yaml"));
    console.log("Desktop runtime smoke passed");
  } finally {
    await cleanupSmokeRuntime(runtime, appDataRoot);
  }
}

export async function cleanupSmokeRuntime(runtime: DesktopRuntime | null, appDataRoot: string): Promise<void> {
  let stopError: unknown;

  try {
    if (runtime) {
      await runtime.stop();
    }
  } catch (error) {
    stopError = error;
  } finally {
    await rm(appDataRoot, { recursive: true, force: true });
  }

  if (stopError) {
    throw stopError;
  }
}

async function requestJson(
  url: string,
  options: {
    method?: string;
    cookies: CookieJar;
    body?: unknown;
    tolerateStatuses?: number[];
  },
): Promise<unknown> {
  const headers = new Headers();
  const cookieHeader = serializeCookies(options.cookies);
  const csrfToken = options.cookies.get("csrf_token");

  if (cookieHeader) {
    headers.set("cookie", cookieHeader);
  }
  if (csrfToken) {
    headers.set("x-csrf-token", csrfToken);
  }
  if (options.body !== undefined) {
    headers.set("content-type", "application/json");
  }

  const response = await fetch(url, {
    method: options.method ?? "GET",
    headers,
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    cache: "no-store",
  });

  storeCookies(options.cookies, response.headers);

  const okStatuses = new Set([...(options.tolerateStatuses ?? []), 200]);
  if (!okStatuses.has(response.status)) {
    const body = await response.text();
    throw new Error(`${url} returned HTTP ${response.status}: ${body.slice(0, 500)}`);
  }

  const text = await response.text();
  return text ? JSON.parse(text) : null;
}

async function assertLocalSandboxConfig(configPath: string): Promise<void> {
  const raw = await readFile(configPath, "utf8");
  const config = parse(raw);
  const sandbox = isRecord(config) && isRecord(config.sandbox) ? config.sandbox : {};

  if (sandbox.use !== "deerflow.sandbox.local:LocalSandboxProvider") {
    throw new Error("Runtime config does not use LocalSandboxProvider.");
  }
  if ("container_prefix" in sandbox) {
    throw new Error("Runtime config must not include sandbox.container_prefix.");
  }
}

function storeCookies(cookies: CookieJar, headers: Headers): void {
  const setCookie = getSetCookie(headers);
  for (const cookie of setCookie) {
    const [pair] = cookie.split(";");
    const separator = pair.indexOf("=");
    if (separator > 0) {
      cookies.set(pair.slice(0, separator), pair.slice(separator + 1));
    }
  }
}

function getSetCookie(headers: Headers): string[] {
  const withGetSetCookie = headers as Headers & { getSetCookie?: () => string[] };
  if (typeof withGetSetCookie.getSetCookie === "function") {
    return withGetSetCookie.getSetCookie();
  }

  const header = headers.get("set-cookie");
  return header ? [header] : [];
}

function serializeCookies(cookies: CookieJar): string {
  return [...cookies.entries()].map(([name, value]) => `${name}=${value}`).join("; ");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function main() {
  await smokeRuntime(parseSmokeRuntimeArgs());
}

export function parseSmokeRuntimeArgs(
  args: string[] = process.argv.slice(2),
  scriptPath = fileURLToPath(import.meta.url),
): SmokeRuntimeArgs {
  const packaged = args.includes("--packaged");
  const positionalRepoRoot = args.find((arg) => !arg.startsWith("--"));
  const scriptDir = dirname(scriptPath);
  const defaultAppPath = positionalRepoRoot
    ? resolve(positionalRepoRoot, "desktop", "electron")
    : resolve(scriptDir, "..", "..");
  const appPath = resolve(getArgValue(args, "--app-path") ?? defaultAppPath);
  const resourcesPath = resolve(
    getArgValue(args, "--resources-path") ??
      getArgValue(args, "--resources") ??
      join(appPath, "resources"),
  );

  return {
    appPath,
    resourcesPath,
    packaged,
  };
}

function getArgValue(args: string[], name: string): string | undefined {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
}

if (process.argv[1]?.endsWith("smoke-runtime.js")) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
