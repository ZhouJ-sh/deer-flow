import { describe, expect, test } from "vitest";

import { shouldOpenExternalUrl } from "../src/main/window.js";

describe("desktop window URL policy", () => {
  test("allows only browser-safe external protocols", () => {
    expect(shouldOpenExternalUrl("https://example.com/docs")).toBe(true);
    expect(shouldOpenExternalUrl("http://example.com/docs")).toBe(true);
    expect(shouldOpenExternalUrl("mailto:support@example.com")).toBe(true);
    expect(shouldOpenExternalUrl("file:///Users/example/.ssh/id_rsa")).toBe(false);
    expect(shouldOpenExternalUrl("deerflow://settings")).toBe(false);
    expect(shouldOpenExternalUrl("javascript:alert(1)")).toBe(false);
    expect(shouldOpenExternalUrl("not a url")).toBe(false);
  });
});
