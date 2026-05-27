import { appendFile, readFile } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { URL } from "node:url";

import httpProxy from "http-proxy";

export type DesktopProxyOptions = {
  host: string;
  port: number;
  gatewayOrigin: string;
  nextOrigin: string;
  tokenPath: string;
  internalHeaderValue: string;
  logPath: string;
};

export type DesktopProxy = {
  origin: string;
  close: () => Promise<void>;
};

const DESKTOP_TOKEN_HEADER = "x-deerflow-desktop-token";
const INTERNAL_NEXT_HEADER = "x-deerflow-desktop-internal-next";

export async function startDesktopProxy(options: DesktopProxyOptions): Promise<DesktopProxy> {
  const proxy = httpProxy.createProxyServer({
    changeOrigin: true,
    xfwd: true,
  });

  proxy.on("proxyRes", (proxyResponse) => {
    delete proxyResponse.headers[DESKTOP_TOKEN_HEADER];
  });

  const server = createServer(async (request, response) => {
    try {
      const url = new URL(request.url ?? "/", `http://${options.host}:${options.port}`);
      const route = routeRequest(url, request, options.internalHeaderValue);

      if (route.kind === "not-found") {
        respond(response, 404, "Not found");
        return;
      }

      delete request.headers[DESKTOP_TOKEN_HEADER];

      if (route.kind === "next") {
        proxy.web(request, response, { target: options.nextOrigin });
        return;
      }

      const token = await readDesktopToken(options.tokenPath);
      request.headers[DESKTOP_TOKEN_HEADER] = token;
      request.url = route.pathAndQuery;
      proxy.web(request, response, { target: options.gatewayOrigin });
    } catch (error) {
      await safeLogProxyError(options.logPath, error);
      respond(response, 502, "Bad gateway");
    }
  });

  proxy.on("error", async (error, _request, response) => {
    await safeLogProxyError(options.logPath, error);
    if (response && "writeHead" in response && !response.headersSent) {
      respond(response, 502, "Bad gateway");
    } else if (response && "end" in response) {
      response.end();
    }
  });

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(options.port, options.host, () => {
      server.off("error", reject);
      resolve();
    });
  });

  return {
    origin: `http://${options.host}:${options.port}`,
    close: () =>
      new Promise<void>((resolve, reject) => {
        proxy.close();
        server.close((error) => (error ? reject(error) : resolve()));
      }),
  };
}

type Route =
  | { kind: "gateway"; pathAndQuery: string }
  | { kind: "next" }
  | { kind: "not-found" };

function routeRequest(
  url: URL,
  request: IncomingMessage,
  internalHeaderValue: string,
): Route {
  if (url.pathname === "/_desktop-gateway" || url.pathname.startsWith("/_desktop-gateway/")) {
    if (request.headers[INTERNAL_NEXT_HEADER] !== internalHeaderValue) {
      return { kind: "not-found" };
    }

    return {
      kind: "gateway",
      pathAndQuery: stripPrefix(url, "/_desktop-gateway"),
    };
  }

  if (url.pathname === "/api/langgraph" || url.pathname.startsWith("/api/langgraph/")) {
    return {
      kind: "gateway",
      pathAndQuery: `/api${url.pathname.slice("/api/langgraph".length)}${url.search}`,
    };
  }

  if (url.pathname === "/api" || url.pathname.startsWith("/api/")) {
    return {
      kind: "gateway",
      pathAndQuery: `${url.pathname}${url.search}`,
    };
  }

  return { kind: "next" };
}

function stripPrefix(url: URL, prefix: string) {
  const path = url.pathname.slice(prefix.length) || "/";
  return `${path}${url.search}`;
}

async function readDesktopToken(tokenPath: string) {
  return (await readFile(tokenPath, "utf8")).trim();
}

async function logProxyError(logPath: string, error: unknown) {
  const message = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  await appendFile(logPath, `[${new Date().toISOString()}] Proxy error: ${message}\n`, "utf8");
}

async function safeLogProxyError(logPath: string, error: unknown) {
  try {
    await logProxyError(logPath, error);
  } catch {
    // Keep proxy failures contained even when the log path is unavailable.
  }
}

function respond(response: ServerResponse, statusCode: number, body: string) {
  response.writeHead(statusCode, {
    "content-type": "text/plain; charset=utf-8",
    "cache-control": "no-store",
  });
  response.end(body);
}
