import { access, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";

import { describe, expect, test } from "vitest";

import { runtimeLayout, stageRuntime } from "../src/scripts/build-runtime.js";
import { smokePackaged } from "../src/scripts/smoke-packaged.js";
import { cleanupSmokeRuntime, parseSmokeRuntimeArgs } from "../src/scripts/smoke-runtime.js";
import { parseStagePythonDepsArgs } from "../src/scripts/stage-python-deps.js";

describe("runtimeLayout", () => {
  test("resolves packaged runtime staging paths under the output directory", () => {
    const repoRoot = resolve("/tmp/deer-flow");
    const outDir = resolve("/tmp/deer-flow/desktop/electron/resources");

    expect(runtimeLayout(repoRoot, outDir)).toEqual({
      backend: join(outDir, "backend"),
      frontend: join(outDir, "frontend"),
      pythonRuntime: join(outDir, "runtimes", "python"),
      nodeRuntime: join(outDir, "runtimes", "node"),
      desktopServer: join(outDir, "desktop-server"),
    });
  });

  test("does not stage non-local sandbox resources", () => {
    const layout = runtimeLayout(resolve("/tmp/deer-flow"), resolve("/tmp/resources"));
    const serialized = JSON.stringify(layout).toLowerCase();

    expect(serialized).not.toContain("docker");
    expect(serialized).not.toContain("aio");
    expect(serialized).not.toContain("apple");
    expect(serialized).not.toContain("container");
    expect(serialized).not.toContain("container_prefix");
  });
});

describe("stage-python-deps arguments", () => {
  test("accepts backend directory first and output site-packages directory second", () => {
    expect(parseStagePythonDepsArgs(["/repo/backend", "/tmp/site-packages"])).toEqual({
      backendDir: resolve("/repo/backend"),
      outDir: resolve("/tmp/site-packages"),
    });
  });

  test("requires an explicit output site-packages directory", () => {
    expect(() => parseStagePythonDepsArgs(["/repo/backend"])).toThrow(
      "Usage: node dist/scripts/stage-python-deps.js <backend-dir> <out-site-packages-dir>",
    );
  });
});

describe("stageRuntime", () => {
  test("copies packaged backend, frontend, desktop server, and runtime artifacts", async () => {
    const fixtureRoot = resolve("node_modules", ".tmp", `stage-runtime-${process.pid}-${Date.now()}`);
    const repoRoot = join(fixtureRoot, "repo");
    const outDir = join(fixtureRoot, "resources");
    const runtimeSources = await writeStageRuntimeFixture(repoRoot, fixtureRoot);

    try {
      await writeFile(join(outDir, "stale.txt"), "stale", "utf8").catch(async () => {
        await mkdir(outDir, { recursive: true });
        await writeFile(join(outDir, "stale.txt"), "stale", "utf8");
      });

      await withRuntimeEnv(runtimeSources, async () => {
        const layout = await stageRuntime(repoRoot, outDir);

        await expect(readFile(join(outDir, "stale.txt"), "utf8")).rejects.toThrow();
        await expect(readFile(join(layout.backend, "app", "main.py"), "utf8")).resolves.toBe("app");
        await expect(readFile(join(layout.backend, "site-packages", "deps.txt"), "utf8")).resolves.toBe("deps");
        await expect(readFile(join(layout.frontend, ".next", "standalone", "server.js"), "utf8")).resolves.toBe("server");
        await expect(
          readFile(join(layout.frontend, ".next", "standalone", ".next", "static", "asset.txt"), "utf8"),
        ).resolves.toBe("asset");
        await expect(readFile(join(layout.desktopServer, "next", "register-fetch.js"), "utf8")).resolves.toBe("next");
        await expect(readFile(join(layout.nodeRuntime, "bin", "node"), "utf8")).resolves.toBe("node");
        await expect(readFile(join(layout.pythonRuntime, "bin", "python"), "utf8")).resolves.toBe("python");
      });
    } finally {
      await rm(fixtureRoot, { recursive: true, force: true });
    }
  });

  test("requires operator-supplied runtime directories", async () => {
    await withRuntimeEnv(
      {
        DEER_FLOW_DESKTOP_NODE_RUNTIME_DIR: undefined,
        DEER_FLOW_DESKTOP_PYTHON_RUNTIME_DIR: undefined,
        DEER_FLOW_DESKTOP_PYTHON_SITE_PACKAGES_DIR: undefined,
      },
      async () => {
        await expect(stageRuntime("/repo", "/out")).rejects.toThrow("DEER_FLOW_DESKTOP_NODE_RUNTIME_DIR is required");
      },
    );
  });

  test("rejects runtime env paths that point to files", async () => {
    const fixtureRoot = resolve("node_modules", ".tmp", `stage-runtime-file-${process.pid}-${Date.now()}`);
    const nodeRuntimeFile = join(fixtureRoot, "node-runtime-file");
    const pythonRuntime = join(fixtureRoot, "python-runtime");
    const sitePackages = join(fixtureRoot, "site-packages");

    try {
      await Promise.all([mkdir(pythonRuntime, { recursive: true }), mkdir(sitePackages, { recursive: true })]);
      await writeFile(nodeRuntimeFile, "not a directory", "utf8");

      await withRuntimeEnv(
        {
          DEER_FLOW_DESKTOP_NODE_RUNTIME_DIR: nodeRuntimeFile,
          DEER_FLOW_DESKTOP_PYTHON_RUNTIME_DIR: pythonRuntime,
          DEER_FLOW_DESKTOP_PYTHON_SITE_PACKAGES_DIR: sitePackages,
        },
        async () => {
          await expect(stageRuntime("/repo", "/out")).rejects.toThrow("not a readable directory");
        },
      );
    } finally {
      await rm(fixtureRoot, { recursive: true, force: true });
    }
  });
});

describe("smoke-runtime arguments", () => {
  test("defaults the dev app path to the Electron app root from the built script path", () => {
    const scriptPath = join("/repo", "desktop", "electron", "dist", "scripts", "smoke-runtime.js");

    expect(parseSmokeRuntimeArgs([], scriptPath)).toEqual({
      appPath: resolve("/repo", "desktop", "electron"),
      packaged: false,
      resourcesPath: resolve("/repo", "desktop", "electron", "resources"),
    });
  });

  test("supports the release smoke positional repo root and --resources alias", () => {
    const scriptPath = join("/repo", "desktop", "electron", "dist", "scripts", "smoke-runtime.js");

    expect(parseSmokeRuntimeArgs(["/repo", "--packaged", "--resources", "/tmp/resources"], scriptPath)).toEqual({
      appPath: resolve("/repo", "desktop", "electron"),
      packaged: true,
      resourcesPath: resolve("/tmp/resources"),
    });
  });

  test("removes temporary app data even when runtime shutdown fails", async () => {
    const appDataRoot = resolve("node_modules", ".tmp", `smoke-runtime-cleanup-${process.pid}-${Date.now()}`);
    await mkdir(appDataRoot, { recursive: true });

    await expect(
      cleanupSmokeRuntime(
        {
          proxyOrigin: "http://127.0.0.1:1",
          stop: async () => {
            throw new Error("stop failed");
          },
        },
        appDataRoot,
      ),
    ).rejects.toThrow("stop failed");
    await expect(exists(appDataRoot)).resolves.toBe(false);
  });
});

describe("smokePackaged", () => {
  test("requires platform runtime executable paths instead of runtime directories only", async () => {
    const resourcesPath = resolve(
      "node_modules",
      ".tmp",
      `smoke-packaged-${process.pid}-${Date.now()}`,
    );

    try {
      await writePackagedFixture(resourcesPath, false);

      await expect(smokePackaged(resourcesPath)).rejects.toThrow(runtimeExecutablePathPattern());

      await writeRuntimeExecutables(resourcesPath);
      await expect(smokePackaged(resourcesPath)).resolves.toBeUndefined();
    } finally {
      await rm(resourcesPath, { recursive: true, force: true });
    }
  });
});

async function writeStageRuntimeFixture(
  repoRoot: string,
  fixtureRoot: string,
): Promise<Record<string, string>> {
  const nodeRuntime = join(fixtureRoot, "node-runtime");
  const pythonRuntime = join(fixtureRoot, "python-runtime");
  const sitePackages = join(fixtureRoot, "site-packages");
  const dirs = [
    join(repoRoot, "backend", "app"),
    join(repoRoot, "backend", "packages"),
    join(repoRoot, "frontend", ".next", "standalone"),
    join(repoRoot, "frontend", ".next", "static"),
    join(repoRoot, "frontend", "public"),
    join(repoRoot, "desktop", "electron", "dist", "proxy"),
    join(repoRoot, "desktop", "electron", "dist", "next"),
    join(nodeRuntime, "bin"),
    join(pythonRuntime, "bin"),
    sitePackages,
  ];

  await Promise.all(dirs.map((dir) => mkdir(dir, { recursive: true })));
  await Promise.all([
    writeFile(join(repoRoot, "backend", "app", "main.py"), "app", "utf8"),
    writeFile(join(repoRoot, "backend", "packages", "package.txt"), "package", "utf8"),
    writeFile(join(repoRoot, "backend", "pyproject.toml"), "pyproject", "utf8"),
    writeFile(join(repoRoot, "backend", "uv.lock"), "lock", "utf8"),
    writeFile(join(repoRoot, "config.example.yaml"), "config", "utf8"),
    writeFile(join(repoRoot, "extensions_config.example.json"), "{}", "utf8"),
    writeFile(join(repoRoot, "frontend", ".next", "standalone", "server.js"), "server", "utf8"),
    writeFile(join(repoRoot, "frontend", ".next", "static", "asset.txt"), "asset", "utf8"),
    writeFile(join(repoRoot, "frontend", "public", "public.txt"), "public", "utf8"),
    writeFile(join(repoRoot, "desktop", "electron", "dist", "proxy", "desktop-proxy.js"), "proxy", "utf8"),
    writeFile(join(repoRoot, "desktop", "electron", "dist", "next", "register-fetch.js"), "next", "utf8"),
    writeFile(join(nodeRuntime, "bin", "node"), "node", "utf8"),
    writeFile(join(pythonRuntime, "bin", "python"), "python", "utf8"),
    writeFile(join(sitePackages, "deps.txt"), "deps", "utf8"),
  ]);

  return {
    DEER_FLOW_DESKTOP_NODE_RUNTIME_DIR: nodeRuntime,
    DEER_FLOW_DESKTOP_PYTHON_RUNTIME_DIR: pythonRuntime,
    DEER_FLOW_DESKTOP_PYTHON_SITE_PACKAGES_DIR: sitePackages,
  };
}

async function withRuntimeEnv(
  values: Record<string, string | undefined>,
  run: () => Promise<void>,
): Promise<void> {
  const previous = Object.fromEntries(
    Object.keys(values).map((name) => [name, process.env[name]]),
  );

  for (const [name, value] of Object.entries(values)) {
    if (value === undefined) {
      delete process.env[name];
    } else {
      process.env[name] = value;
    }
  }

  try {
    await run();
  } finally {
    for (const [name, value] of Object.entries(previous)) {
      if (value === undefined) {
        delete process.env[name];
      } else {
        process.env[name] = value;
      }
    }
  }
}

async function writePackagedFixture(resourcesPath: string, includeRuntimeExecutables: boolean) {
  const dirs = [
    join(resourcesPath, "backend", "app"),
    join(resourcesPath, "backend", "packages"),
    join(resourcesPath, "backend", "site-packages"),
    join(resourcesPath, "frontend", ".next", "standalone", ".next", "static"),
    join(resourcesPath, "frontend", ".next", "standalone", "public"),
    join(resourcesPath, "desktop-server", "proxy"),
    join(resourcesPath, "desktop-server", "next"),
    join(resourcesPath, "runtimes", "node"),
    join(resourcesPath, "runtimes", "python"),
  ];

  await Promise.all(dirs.map((dir) => mkdir(dir, { recursive: true })));
  await Promise.all([
    writeFile(join(resourcesPath, "backend", "pyproject.toml"), "", "utf8"),
    writeFile(join(resourcesPath, "backend", "uv.lock"), "", "utf8"),
    writeFile(join(resourcesPath, "backend", "config.example.yaml"), "", "utf8"),
    writeFile(join(resourcesPath, "backend", "extensions_config.example.json"), "{}", "utf8"),
  ]);

  if (includeRuntimeExecutables) {
    await writeRuntimeExecutables(resourcesPath);
  }
}

async function writeRuntimeExecutables(resourcesPath: string) {
  await Promise.all(
    runtimeExecutablePaths(resourcesPath).map(async (path) => {
      await mkdir(dirname(path), { recursive: true });
      await writeFile(path, "", "utf8");
    }),
  );
}

async function exists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

function runtimeExecutablePaths(resourcesPath: string): string[] {
  if (process.platform === "win32") {
    return [
      join(resourcesPath, "runtimes", "node", "node.exe"),
      join(resourcesPath, "runtimes", "python", "python.exe"),
    ];
  }

  return [
    join(resourcesPath, "runtimes", "node", "bin", "node"),
    join(resourcesPath, "runtimes", "python", "bin", "python"),
  ];
}

function runtimeExecutablePathPattern(): RegExp {
  const executablePaths = runtimeExecutablePaths("");
  const missingExecutable = executablePaths[0].startsWith("/") ? executablePaths[0].slice(1) : executablePaths[0];
  return new RegExp(missingExecutable.replaceAll("\\", "\\\\").replaceAll("/", "[/\\\\]"));
}
