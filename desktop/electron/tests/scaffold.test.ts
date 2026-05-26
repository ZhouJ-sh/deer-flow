import { describe, expect, test } from "vitest";

describe("desktop scaffold", () => {
  test("sets the desktop app name", async () => {
    const pkg = await import("../package.json", { assert: { type: "json" } });
    expect(pkg.default.name).toBe("@deer-flow/desktop-electron");
  });

  test("exposes packaged layout smoke script after smoke packaging exists", async () => {
    const pkg = await import("../package.json", { assert: { type: "json" } });
    expect(pkg.default.scripts.smoke).toBe("node dist/scripts/smoke-packaged.js");
  });

  test("builds before packaging from a clean output directory", async () => {
    const pkg = await import("../package.json", { assert: { type: "json" } });
    expect(pkg.default.scripts.pack).toBe("pnpm build && electron-builder --dir");
    expect(pkg.default.scripts.dist).toBe("pnpm build && electron-builder");
  });

  test("loads the bundled preload entrypoint from the desktop window", async () => {
    const window = await import("node:fs/promises").then(({ readFile }) =>
      readFile(new URL("../src/main/window.ts", import.meta.url), "utf8"),
    );

    expect(window).toContain("preload:");
    expect(window).toContain('join(__dirname, "..", "preload", "index.js")');
  });

  test("emits main as ESM and preload as CommonJS", async () => {
    const { default: tsupConfig } = await import("../tsup.config.js");

    expect(Array.isArray(tsupConfig)).toBe(true);
    const configs = tsupConfig as Array<{
      entry?: Record<string, string>;
      format?: string[];
      outDir?: string;
      outExtension?: () => { js?: string };
    }>;
    const mainConfig = configs.find((config) => config.entry?.["main/index"]);
    const preloadConfig = configs.find((config) => config.entry?.["preload/index"]);

    expect(mainConfig).toMatchObject({
      entry: {
        "main/index": "src/main/index.ts",
        "next/register-fetch": "src/next/register-fetch.ts",
        "proxy/desktop-proxy": "src/proxy/desktop-proxy.ts",
        "scripts/build-runtime": "src/scripts/build-runtime.ts",
        "scripts/smoke-packaged": "src/scripts/smoke-packaged.ts",
        "scripts/smoke-runtime": "src/scripts/smoke-runtime.ts",
        "scripts/stage-python-deps": "src/scripts/stage-python-deps.ts",
      },
      format: ["esm"],
      outDir: "dist",
    });
    expect(mainConfig?.entry).not.toHaveProperty("preload/index");
    expect(preloadConfig).toMatchObject({
      entry: { "preload/index": "src/preload/index.ts" },
      format: ["cjs"],
      outDir: "dist",
    });
    expect(preloadConfig?.outExtension?.().js).toBe(".js");
    expect(preloadConfig?.entry).not.toHaveProperty("main/index");
  });

  test("packages generated runtime resources after staging support exists", async () => {
    const builderConfig = await import("node:fs/promises").then(({ readFile }) =>
      readFile(new URL("../electron-builder.yml", import.meta.url), "utf8"),
    );

    expect(builderConfig).toContain("extraResources:");
    expect(builderConfig).toContain("from: resources");
    expect(builderConfig).toContain("to: .");
    expect(builderConfig).toContain('"**/*"');
  });
});
