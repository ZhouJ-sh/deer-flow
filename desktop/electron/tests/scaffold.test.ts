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

  test("loads the bundled preload entrypoint from the main process", async () => {
    const main = await import("node:fs/promises").then(({ readFile }) =>
      readFile(new URL("../src/main/index.ts", import.meta.url), "utf8"),
    );

    expect(main).toContain("preload:");
    expect(main).toContain('join(__dirname, "..", "preload", "index.js")');
  });

  test("does not reference generated resources before they exist", async () => {
    const builderConfig = await import("node:fs/promises").then(({ readFile }) =>
      readFile(new URL("../electron-builder.yml", import.meta.url), "utf8"),
    );

    expect(builderConfig).not.toContain("extraResources:");
  });
});
