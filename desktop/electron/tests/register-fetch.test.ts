import { afterEach, describe, expect, test, vi } from "vitest";

const originalEnv = { ...process.env };
const originalFetch = globalThis.fetch;

afterEach(() => {
  process.env = { ...originalEnv };
  globalThis.fetch = originalFetch;
  vi.resetModules();
});

async function loadRegisterFetch() {
  await import("../src/next/register-fetch.js");
}

describe("register-fetch", () => {
  test("adds the internal header only for the desktop gateway path", async () => {
    process.env.DEER_FLOW_INTERNAL_GATEWAY_BASE_URL = "http://127.0.0.1:5000/_desktop-gateway";
    process.env.DEER_FLOW_INTERNAL_GATEWAY_HEADER_NAME = "x-internal";
    process.env.DEER_FLOW_INTERNAL_GATEWAY_HEADER_VALUE = "secret";
    const requests: Request[] = [];
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      requests.push(new Request(input, init));
      return new Response("ok");
    });

    await loadRegisterFetch();
    await fetch("http://127.0.0.1:5000/_desktop-gateway/api/models");
    await fetch("http://127.0.0.1:5000/_desktop-gatewayfoo/api/models");

    expect(requests[0].headers.get("x-internal")).toBe("secret");
    expect(requests[1].headers.get("x-internal")).toBeNull();
  });

  test("preserves existing Request headers when marking internal fetches", async () => {
    process.env.DEER_FLOW_INTERNAL_GATEWAY_BASE_URL = "http://127.0.0.1:5000/_desktop-gateway";
    process.env.DEER_FLOW_INTERNAL_GATEWAY_HEADER_NAME = "x-internal";
    process.env.DEER_FLOW_INTERNAL_GATEWAY_HEADER_VALUE = "secret";
    let request: Request | undefined;
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      request = new Request(input, init);
      return new Response("ok");
    });

    await loadRegisterFetch();
    await fetch(
      new Request("http://127.0.0.1:5000/_desktop-gateway/api/models", {
        headers: { "x-existing": "kept" },
      }),
    );

    if (!request) {
      throw new Error("Expected fetch to be called");
    }
    expect(request.headers.get("x-existing")).toBe("kept");
    expect(request.headers.get("x-internal")).toBe("secret");
  });
});
