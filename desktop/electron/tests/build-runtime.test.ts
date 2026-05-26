import { join, resolve } from "node:path";

import { describe, expect, test } from "vitest";

import { runtimeLayout } from "../src/scripts/build-runtime.js";

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
