import { join, resolve } from "node:path";

import { describe, expect, test } from "vitest";

import { resolveDesktopResources } from "../src/main/paths.js";

describe("resolveDesktopResources", () => {
  test("resolves development paths from appPath without relying on cwd", () => {
    const repoRoot = resolve("/tmp/deer-flow");
    const appPath = join(repoRoot, "desktop", "electron");

    const paths = resolveDesktopResources({
      packaged: false,
      appPath,
      resourcesPath: "/unused/resources",
    });

    expect(paths).toEqual({
      repoRoot,
      backendDir: join(repoRoot, "backend"),
      frontendDir: join(repoRoot, "frontend"),
      pythonBin: "python",
      nodeBin: "node",
      desktopServerDir: join(appPath, "dist"),
    });
  });

  test("resolves packaged unix runtime paths from resourcesPath", () => {
    const resourcesPath = resolve("/opt/DeerFlow/resources");

    const paths = resolveDesktopResources({
      packaged: true,
      appPath: "/unused/app",
      resourcesPath,
      platform: "darwin",
    });

    expect(paths).toEqual({
      repoRoot: null,
      backendDir: join(resourcesPath, "backend"),
      frontendDir: join(resourcesPath, "frontend"),
      pythonBin: join(resourcesPath, "runtimes", "python", "bin", "python"),
      nodeBin: join(resourcesPath, "runtimes", "node", "bin", "node"),
      desktopServerDir: join(resourcesPath, "desktop-server"),
    });
  });

  test("resolves packaged win32 runtime paths from resourcesPath", () => {
    const resourcesPath = resolve("/opt/DeerFlow/resources");

    const paths = resolveDesktopResources({
      packaged: true,
      appPath: "/unused/app",
      resourcesPath,
      platform: "win32",
    });

    expect(paths.pythonBin).toBe(join(resourcesPath, "runtimes", "python", "python.exe"));
    expect(paths.nodeBin).toBe(join(resourcesPath, "runtimes", "node", "node.exe"));
  });
});
