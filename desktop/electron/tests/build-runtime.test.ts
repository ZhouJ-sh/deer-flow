import { mkdir, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";

import { describe, expect, test } from "vitest";

import { runtimeLayout } from "../src/scripts/build-runtime.js";
import { smokePackaged } from "../src/scripts/smoke-packaged.js";
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
