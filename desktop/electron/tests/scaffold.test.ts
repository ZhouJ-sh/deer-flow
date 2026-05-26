import { describe, expect, test } from "vitest";

describe("desktop scaffold", () => {
  test("sets the desktop app name", async () => {
    const pkg = await import("../package.json", { assert: { type: "json" } });
    expect(pkg.default.name).toBe("@deer-flow/desktop-electron");
  });

  test("does not expose a smoke script before smoke packaging exists", async () => {
    const pkg = await import("../package.json", { assert: { type: "json" } });
    expect(pkg.default.scripts).not.toHaveProperty("smoke");
  });

  test("builds before packaging from a clean output directory", async () => {
    const pkg = await import("../package.json", { assert: { type: "json" } });
    expect(pkg.default.scripts.pack).toBe("pnpm build && electron-builder --dir");
    expect(pkg.default.scripts.dist).toBe("pnpm build && electron-builder");
  });

  test("loads the bundled preload entrypoint from the main process", async () => {
    const main = await import("node:fs/promises").then(({ readFile }) =>
      readFile(new URL("../src/main/index.ts", import.meta.url), "utf8"),
    );

    expect(main).toContain("preload:");
    expect(main).toContain('join(__dirname, "..", "preload", "index.js")');
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
      entry: { "main/index": "src/main/index.ts" },
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

  test("does not reference generated resources before they exist", async () => {
    const builderConfig = await import("node:fs/promises").then(({ readFile }) =>
      readFile(new URL("../electron-builder.yml", import.meta.url), "utf8"),
    );

    expect(builderConfig).not.toContain("extraResources:");
  });
});
