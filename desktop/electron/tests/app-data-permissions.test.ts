import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, test, vi } from "vitest";

const { chmodFailure } = vi.hoisted(() => ({
  chmodFailure: Object.assign(new Error("chmod denied"), { code: "EPERM" }),
}));

vi.mock("node:fs/promises", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs/promises")>();
  return {
    ...actual,
    chmod: vi.fn(async () => {
      throw chmodFailure;
    }),
  };
});

const { ensureSecretTextFile } = await import("../src/main/app-data.js");

let created: string[] = [];

afterEach(async () => {
  await Promise.all(created.map((dir) => rm(dir, { recursive: true, force: true })));
  created = [];
});

async function makeRoot() {
  const root = await mkdtemp(join(tmpdir(), "deer-flow-desktop-permissions-"));
  created.push(root);
  return root;
}

describe("desktop secret file permissions", () => {
  test.skipIf(process.platform === "win32")("surfaces chmod failures for secret files on POSIX", async () => {
    const root = await makeRoot();

    await expect(ensureSecretTextFile(join(root, "desktop-token"), () => "secret\n")).rejects.toThrow(
      "chmod denied",
    );
  });
});
