import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import getPort from "get-port";
import { afterEach, describe, expect, test } from "vitest";

import { startDesktopProxy } from "../src/proxy/desktop-proxy.js";

type CapturedRequest = {
  method: string | undefined;
  url: string | undefined;
  desktopToken: string | undefined;
};

type TestServer = {
  origin: string;
  requests: CapturedRequest[];
  close: () => Promise<void>;
};

let servers: TestServer[] = [];
let dirs: string[] = [];
let proxies: Array<{ close: () => Promise<void> }> = [];

afterEach(async () => {
  await Promise.allSettled(proxies.map((proxy) => proxy.close()));
  proxies = [];
  await Promise.allSettled(servers.map((server) => server.close()));
  servers = [];
  await Promise.allSettled(dirs.map((dir) => rm(dir, { recursive: true, force: true })));
  dirs = [];
});

async function tempDir() {
  const dir = await mkdtemp(join(tmpdir(), "deer-flow-proxy-"));
  dirs.push(dir);
  return dir;
}

async function createTokenFile(value = " desktop-secret-token \n") {
  const path = join(await tempDir(), "token");
  await writeFile(path, value, "utf8");
  return path;
}

async function startRecordingServer(
  handler?: (request: IncomingMessage, response: ServerResponse) => void,
): Promise<TestServer> {
  const requests: CapturedRequest[] = [];
  const server = createServer((request, response) => {
    requests.push({
      method: request.method,
      url: request.url,
      desktopToken: request.headers["x-deerflow-desktop-token"] as string | undefined,
    });
    if (handler) {
      handler(request, response);
      return;
    }
    response.setHeader("content-type", "application/json");
    response.end(JSON.stringify({ ok: true, url: request.url }));
  });

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });

  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("Expected TCP server address");
  }

  return {
    origin: `http://127.0.0.1:${address.port}`,
    requests,
    close: () =>
      new Promise<void>((resolve, reject) => {
        server.close((error) => (error ? reject(error) : resolve()));
      }),
  };
}

async function startTestProxy(options?: {
  gatewayOrigin?: string;
  nextOrigin?: string;
  tokenPath?: string;
  logPath?: string;
}) {
  const gateway = options?.gatewayOrigin ? undefined : await startRecordingServer();
  const next = options?.nextOrigin ? undefined : await startRecordingServer();
  if (gateway) servers.push(gateway);
  if (next) servers.push(next);

  const proxy = await startDesktopProxy({
    host: "127.0.0.1",
    port: await getPort({ host: "127.0.0.1" }),
    gatewayOrigin: options?.gatewayOrigin ?? gateway!.origin,
    nextOrigin: options?.nextOrigin ?? next!.origin,
    tokenPath: options?.tokenPath ?? (await createTokenFile()),
    internalHeaderValue: "internal-secret",
    logPath: options?.logPath ?? join(await tempDir(), "proxy.log"),
  });
  proxies.push(proxy);

  return { proxy, gateway, next };
}

describe("startDesktopProxy", () => {
  test("injects the trimmed desktop token for /api/* gateway requests", async () => {
    const { proxy, gateway } = await startTestProxy();

    const response = await fetch(`${proxy.origin}/api/v1/projects`);

    expect(response.status).toBe(200);
    expect(gateway!.requests).toMatchObject([
      { url: "/api/v1/projects", desktopToken: "desktop-secret-token" },
    ]);
  });

  test("maps /api/langgraph/runs query requests to gateway /api/runs", async () => {
    const { proxy, gateway } = await startTestProxy();

    const response = await fetch(`${proxy.origin}/api/langgraph/runs?stream=true`);

    expect(response.status).toBe(200);
    expect(gateway!.requests).toMatchObject([
      { url: "/api/runs?stream=true", desktopToken: "desktop-secret-token" },
    ]);
  });

  test("maps bare /api/langgraph query requests to gateway /api", async () => {
    const { proxy, gateway } = await startTestProxy();

    const response = await fetch(`${proxy.origin}/api/langgraph?foo=bar`);

    expect(response.status).toBe(200);
    expect(gateway!.requests).toMatchObject([
      { url: "/api?foo=bar", desktopToken: "desktop-secret-token" },
    ]);
  });

  test("proxies non-API requests to Next without leaking reserved desktop token headers", async () => {
    const { proxy, gateway, next } = await startTestProxy();

    const response = await fetch(`${proxy.origin}/dashboard?tab=home`, {
      headers: {
        "x-deerflow-desktop-token": "browser-forged-token",
      },
    });

    expect(response.status).toBe(200);
    expect(gateway!.requests).toHaveLength(0);
    expect(next!.requests).toMatchObject([{ url: "/dashboard?tab=home", desktopToken: undefined }]);
  });

  test("rejects browser access to /_desktop-gateway/* with 404", async () => {
    const { proxy, gateway } = await startTestProxy();

    const response = await fetch(`${proxy.origin}/_desktop-gateway/api/v1/auth/setup-status`, {
      headers: {
        "x-deerflow-desktop-internal-next": "wrong-secret",
      },
    });

    expect(response.status).toBe(404);
    expect(await response.text()).not.toContain("desktop-secret-token");
    expect(gateway!.requests).toHaveLength(0);
  });

  test("allows internal /_desktop-gateway/* requests with the exact internal header", async () => {
    const { proxy, gateway } = await startTestProxy();

    const response = await fetch(`${proxy.origin}/_desktop-gateway/api/v1/auth/setup-status`, {
      headers: {
        "x-deerflow-desktop-internal-next": "internal-secret",
      },
    });

    expect(response.status).toBe(200);
    expect(gateway!.requests).toMatchObject([
      { url: "/api/v1/auth/setup-status", desktopToken: "desktop-secret-token" },
    ]);
  });

  test("returns 502 and logs an error when the desktop token cannot be read", async () => {
    const logPath = join(await tempDir(), "proxy-error.log");
    const missingTokenPath = join(await tempDir(), "missing-token");
    const { proxy, gateway } = await startTestProxy({ tokenPath: missingTokenPath, logPath });

    const response = await fetch(`${proxy.origin}/api/v1/projects`);

    expect(response.status).toBe(502);
    expect(await response.text()).not.toContain("desktop-secret-token");
    expect(gateway!.requests).toHaveLength(0);
    await expect(readFile(logPath, "utf8")).resolves.toContain("Proxy error");
  });

  test("returns 502 and logs an error when proxying to Gateway fails", async () => {
    const logPath = join(await tempDir(), "proxy-error.log");
    const closedGateway = await startRecordingServer();
    await closedGateway.close();
    const { proxy } = await startTestProxy({ gatewayOrigin: closedGateway.origin, logPath });

    const response = await fetch(`${proxy.origin}/api/v1/projects`);

    expect(response.status).toBe(502);
    expect(await response.text()).not.toContain("desktop-secret-token");
    await expect(readFile(logPath, "utf8")).resolves.toContain("Proxy error");
  });
});
