import { join, resolve } from "node:path";

import { describe, expect, test } from "vitest";

import {
  buildGatewayCommand,
  buildNextCommand,
  classifyReadinessFailure,
  cleanupAfterStartupFailure,
  resolveDesktopDataSources,
  stopRuntime,
} from "../src/main/runtime.js";
import { resolveDesktopResources } from "../src/main/paths.js";

describe("runtime command builders", () => {
  test("builds a development gateway command using uvicorn on loopback", () => {
    const backendDir = resolve("/repo/backend");
    const dataRoot = resolve("/user/data");

    const command = buildGatewayCommand({
      packaged: false,
      backendDir,
      pythonBin: "/unused/python",
      port: 4321,
      dataRoot,
    });

    expect(command.command).toBe("uv");
    expect(command.cwd).toBe(dataRoot);
    expect(command.args).toEqual([
      "--project",
      backendDir,
      "run",
      "uvicorn",
      "app.gateway.app:app",
      "--host",
      "127.0.0.1",
      "--port",
      "4321",
    ]);
    expect(command.env.GATEWAY_PORT).toBe("4321");
    expect(command.env.PYTHONPATH).toContain(backendDir);
    expect(command.env.PYTHONPATH).toContain(join(backendDir, "packages", "harness"));
  });

  test("builds a packaged gateway command with packaged Python paths", () => {
    const backendDir = resolve("/resources/backend");
    const dataRoot = resolve("/user/data");

    const command = buildGatewayCommand({
      packaged: true,
      backendDir,
      pythonBin: "/resources/runtimes/python/bin/python",
      port: 4322,
      dataRoot,
    });

    expect(command.command).toBe("/resources/runtimes/python/bin/python");
    expect(command.cwd).toBe(dataRoot);
    expect(command.args).toEqual([
      "-m",
      "uvicorn",
      "app.gateway.app:app",
      "--host",
      "127.0.0.1",
      "--port",
      "4322",
    ]);
    expect(command.env.GATEWAY_PORT).toBe("4322");
    expect(command.env.PYTHONPATH).toContain(backendDir);
    expect(command.env.PYTHONPATH).toContain(join(backendDir, "packages", "harness"));
    expect(command.env.PYTHONPATH).toContain(join(backendDir, "site-packages"));
  });

  test("builds a packaged Next standalone command with register fetch preloaded", () => {
    const frontendDir = resolve("/resources/frontend");
    const registerFetchPath = resolve("/resources/desktop-server/next/register-fetch.js");

    const command = buildNextCommand({
      packaged: true,
      frontendDir,
      nodeBin: "/resources/runtimes/node/bin/node",
      port: 5432,
      registerFetchPath,
    });

    expect(command.command).toBe("/resources/runtimes/node/bin/node");
    expect(command.cwd).toBe(frontendDir);
    expect(command.args).toEqual([
      "--import",
      registerFetchPath,
      join(frontendDir, ".next", "standalone", "server.js"),
    ]);
    expect(command.env.PORT).toBe("5432");
    expect(command.env.HOSTNAME).toBe("127.0.0.1");
  });

  test("builds a development Next command with SSR fetch marker in NODE_OPTIONS", () => {
    const frontendDir = resolve("/repo/frontend");
    const registerFetchPath = resolve("/repo/desktop/electron/dist/next/register-fetch.js");

    const command = buildNextCommand({
      packaged: false,
      frontendDir,
      nodeBin: "/unused/node",
      port: 6543,
      registerFetchPath,
    });

    expect(command.command).toBe("pnpm");
    expect(command.cwd).toBe(frontendDir);
    expect(command.args).toEqual(["dev", "--hostname", "127.0.0.1", "--port", "6543"]);
    expect(command.env.PORT).toBe("6543");
    expect(command.env.HOSTNAME).toBe("127.0.0.1");
    expect(command.env.NODE_OPTIONS).toContain("--import");
    expect(command.env.NODE_OPTIONS).toContain(registerFetchPath);
  });
});

describe("runtime resource resolution", () => {
  test("development desktop server bundle dir resolves to desktop/electron/dist", () => {
    const repoRoot = resolve("/repo/deer-flow");
    const appPath = join(repoRoot, "desktop", "electron");

    const resources = resolveDesktopResources({
      packaged: false,
      appPath,
      resourcesPath: "/unused",
    });

    expect(resources.desktopServerDir).toBe(join(appPath, "dist"));
  });
});

describe("desktop data source resolution", () => {
  test("uses the repository config as the development source config", () => {
    const repoRoot = resolve("/repo/deer-flow");
    const backendDir = join(repoRoot, "backend");

    const sources = resolveDesktopDataSources({
      packaged: false,
      repoRoot,
      backendDir,
      resourcesPath: "/unused",
    });

    expect(sources.exampleConfigPath).toBe(join(repoRoot, "config.yaml"));
    expect(sources.logsDir).toBe(join(repoRoot, "logs"));
    expect(sources.syncConfigModelsFromSource).toBe(true);
  });

  test("keeps packaged mode on bundled backend example config", () => {
    const resourcesPath = resolve("/resources");
    const backendDir = join(resourcesPath, "backend");

    const sources = resolveDesktopDataSources({
      packaged: true,
      repoRoot: null,
      backendDir,
      resourcesPath,
    });

    expect(sources.exampleConfigPath).toBe(join(backendDir, "config.example.yaml"));
    expect(sources.logsDir).toBeUndefined();
    expect(sources.syncConfigModelsFromSource).toBe(false);
  });
});

describe("readiness failure classification", () => {
  test("produces actionable messages for common startup failures", () => {
    expect(classifyReadinessFailure("gateway", new Error("timeout"), "ModuleNotFoundError: app")).toContain(
      "Gateway failed to import",
    );
    expect(classifyReadinessFailure("gateway", new Error("timeout"), "yaml validation failed")).toContain(
      "config invalid",
    );
    expect(classifyReadinessFailure("gateway", new Error("timeout"), "missing model api key")).toContain(
      "model key missing",
    );
    expect(classifyReadinessFailure("frontend", new Error("timeout"), "next server failed")).toContain(
      "frontend failed to start",
    );
    expect(
      classifyReadinessFailure("gateway", new Error("timeout"), "LocalSandboxProvider allow_host_bash sandbox"),
    ).toContain("local sandbox configuration invalid");
  });

  test("prefers specific import diagnostics over broad sandbox text", () => {
    expect(
      classifyReadinessFailure(
        "gateway",
        new Error("timeout"),
        "ModuleNotFoundError: No module named 'sandbox.plugins'",
      ),
    ).toContain("Gateway failed to import");
  });
});

describe("runtime shutdown", () => {
  test("closes proxy before stopping sidecars in reverse launch order", async () => {
    const calls: string[] = [];
    const proxy = {
      close: async () => {
        calls.push("proxy");
      },
    };
    const gateway = {
      stop: async () => {
        calls.push("gateway");
      },
    };
    const next = {
      stop: async () => {
        calls.push("next");
      },
    };

    await stopRuntime(proxy, next, gateway);

    expect(calls).toEqual(["proxy", "next", "gateway"]);
  });

  test("attempts every cleanup step even when one component fails", async () => {
    const calls: string[] = [];
    const proxy = {
      close: async () => {
        calls.push("proxy");
        throw new Error("proxy close failed");
      },
    };
    const gateway = {
      stop: async () => {
        calls.push("gateway");
      },
    };
    const next = {
      stop: async () => {
        calls.push("next");
      },
    };

    await expect(stopRuntime(proxy, next, gateway)).rejects.toThrow("runtime cleanup failed");

    expect(calls).toEqual(["proxy", "next", "gateway"]);
  });

  test("attempts sidecar cleanup when proxy close hangs", async () => {
    const calls: string[] = [];
    const proxy = {
      close: async () => {
        calls.push("proxy");
        await new Promise(() => {
          // Intentionally never settles.
        });
      },
    };
    const gateway = {
      stop: async () => {
        calls.push("gateway");
      },
    };
    const next = {
      stop: async () => {
        calls.push("next");
      },
    };

    await expect(stopRuntime(proxy, next, gateway, 5)).rejects.toThrow("runtime cleanup failed");

    expect(calls).toEqual(["proxy", "next", "gateway"]);
  });

  test("preserves startup failure details when cleanup also fails", async () => {
    const startupError = new Error("Gateway failed to import");
    const proxy = {
      close: async () => {
        throw new Error("proxy close failed");
      },
    };

    const error = await cleanupAfterStartupFailure(startupError, proxy, null, null);

    expect(error).toBeInstanceOf(AggregateError);
    expect((error as AggregateError).message).toBe("desktop runtime startup failed during cleanup");
    expect((error as AggregateError).errors[0]).toBe(startupError);
    expect((error as AggregateError).errors[1]).toBeInstanceOf(AggregateError);
  });
});
