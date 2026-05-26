import { describe, expect, test } from "vitest";

describe("desktop scaffold", () => {
  test("sets the desktop app name", async () => {
    const pkg = await import("../package.json", { assert: { type: "json" } });
    expect(pkg.default.name).toBe("@deer-flow/desktop-electron");
  });
});
