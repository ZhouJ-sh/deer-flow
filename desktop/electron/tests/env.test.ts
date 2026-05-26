import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, test } from "vitest";

import { ensureDesktopData } from "../src/main/app-data.js";
import { buildGatewayEnv, buildNextEnv, loadDesktopDotEnv } from "../src/main/env.js";

async function makeRoot() {
  return mkdtemp(join(tmpdir(), "deer-flow-desktop-env-"));
}

const originalEnv = { ...process.env };

afterEach(() => {
  process.env = { ...originalEnv };
});

describe("desktop env helpers", () => {
  test("parses app-data .env without mutating process.env", async () => {
    const root = await makeRoot();
    const paths = await ensureDesktopData({ root });
    process.env.DESKTOP_ENV_ONLY = "from-process";
    await writeFile(
      paths.envPath,
      ["DESKTOP_ENV_ONLY=from-dotenv", "SPACED_VALUE=\"hello world\"", "EMPTY_VALUE=", ""].join("\n"),
    );

    const parsed = await loadDesktopDotEnv(paths.envPath);

    expect(parsed).toEqual({
      DESKTOP_ENV_ONLY: "from-dotenv",
      SPACED_VALUE: "hello world",
      EMPTY_VALUE: "",
    });
    expect(process.env.DESKTOP_ENV_ONLY).toBe("from-process");
    expect(process.env.SPACED_VALUE).toBeUndefined();
  });

  test("builds gateway env from process, dotenv, and desktop overrides", async () => {
    const root = await makeRoot();
    const paths = await ensureDesktopData({ root });
    process.env.OPENAI_API_KEY = "from-process";
    process.env.DEER_FLOW_DESKTOP = "0";
    await writeFile(paths.envPath, ["OPENAI_API_KEY=from-dotenv", "CUSTOM_FROM_DOTENV=kept", ""].join("\n"));

    const env = await buildGatewayEnv(paths, { frontendOrigin: "app://deer-flow" });

    expect(env.OPENAI_API_KEY).toBe("from-dotenv");
    expect(env.CUSTOM_FROM_DOTENV).toBe("kept");
    expect(env.DEER_FLOW_DESKTOP).toBe("1");
    expect(env.DEER_FLOW_HOME).toBe(paths.deerFlowHome);
    expect(env.DEER_FLOW_CONFIG_PATH).toBe(paths.configPath);
    expect(env.DEER_FLOW_EXTENSIONS_CONFIG_PATH).toBe(paths.extensionsConfigPath);
    expect(env.DEER_FLOW_PROJECT_ROOT).toBe(paths.root);
    expect(env.DEER_FLOW_SKILLS_PATH).toBe(join(paths.root, "skills"));
    expect(env.DEER_FLOW_DESKTOP_TOKEN_FILE).toBe(paths.tokenPath);
    expect(env.GATEWAY_HOST).toBe("127.0.0.1");
    expect(env.GATEWAY_CORS_ORIGINS).toBe("app://deer-flow");
  });

  test("builds next env with persisted auth secret and no public backend URLs", async () => {
    const root = await makeRoot();
    const paths = await ensureDesktopData({ root });
    process.env.NEXT_PUBLIC_LANGGRAPH_BASE_URL = "http://process-langgraph";
    process.env.NEXT_PUBLIC_BACKEND_BASE_URL = "http://process-backend";
    process.env.BETTER_AUTH_SECRET = "process-secret";
    await writeFile(
      paths.envPath,
      [
        "NEXT_PUBLIC_LANGGRAPH_BASE_URL=http://dotenv-langgraph",
        "NEXT_PUBLIC_BACKEND_BASE_URL=http://dotenv-backend",
        "BETTER_AUTH_SECRET=dotenv-secret",
        "CUSTOM_NEXT_VALUE=kept",
        "",
      ].join("\n"),
    );

    const env = await buildNextEnv(paths, { proxyOrigin: "http://127.0.0.1:5123" });
    const persisted = await readFile(join(paths.deerFlowHome, "better-auth-secret"), "utf8");

    expect(env.CUSTOM_NEXT_VALUE).toBe("kept");
    expect(env.BETTER_AUTH_SECRET).toBe(persisted.trim());
    expect(env.BETTER_AUTH_SECRET).not.toBe("process-secret");
    expect(env.BETTER_AUTH_SECRET).not.toBe("dotenv-secret");
    expect(env.BETTER_AUTH_SECRET).toMatch(/^[A-Za-z0-9_-]{43,}$/);
    expect(env.DEER_FLOW_INTERNAL_GATEWAY_BASE_URL).toBe("http://127.0.0.1:5123/_desktop-gateway");
    expect(env.DEER_FLOW_TRUSTED_ORIGINS).toBe("http://127.0.0.1:5123");
    expect(env.SKIP_ENV_VALIDATION).toBe("1");
    expect(env.NEXT_PUBLIC_LANGGRAPH_BASE_URL).toBeUndefined();
    expect(env.NEXT_PUBLIC_BACKEND_BASE_URL).toBeUndefined();

    const second = await buildNextEnv(paths, { proxyOrigin: "http://127.0.0.1:5124" });
    expect(second.BETTER_AUTH_SECRET).toBe(env.BETTER_AUTH_SECRET);
    await expect(readFile(join(paths.deerFlowHome, "better-auth-secret"), "utf8")).resolves.toBe(persisted);
  });
});
