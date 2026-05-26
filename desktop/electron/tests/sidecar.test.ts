import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, test } from "vitest";

import { startSidecar } from "../src/main/sidecar.js";

async function tempLogPath(name: string) {
  const dir = await mkdtemp(join(tmpdir(), "deer-flow-sidecar-"));
  return join(dir, name);
}

describe("startSidecar", () => {
  test("captures stdout and stderr in the configured log file", async () => {
    const logPath = await tempLogPath("output.log");
    const sidecar = startSidecar({
      name: "log-writer",
      command: process.execPath,
      args: ["-e", "console.log('from stdout'); console.error('from stderr');"],
      cwd: process.cwd(),
      env: {},
      logPath,
    });

    await expect(sidecar.exit).resolves.toBe(0);
    await expect(readFile(logPath, "utf8")).resolves.toContain("from stdout");
    await expect(readFile(logPath, "utf8")).resolves.toContain("from stderr");
  });

  test("rejects exit when the process cannot be spawned", async () => {
    const logPath = await tempLogPath("spawn-error.log");
    const sidecar = startSidecar({
      name: "missing-command",
      command: "definitely-not-a-real-deer-flow-command",
      args: [],
      cwd: process.cwd(),
      env: {},
      logPath,
    });

    await expect(sidecar.exit).rejects.toMatchObject({ code: "ENOENT" });
  });

  test("stop terminates a running process and is a no-op after exit", async () => {
    const logPath = await tempLogPath("stop.log");
    const sidecar = startSidecar({
      name: "long-running",
      command: process.execPath,
      args: ["-e", "setInterval(() => {}, 1000);"],
      cwd: process.cwd(),
      env: {},
      logPath,
    });

    await expect(sidecar.stop()).resolves.toBeUndefined();
    await expect(sidecar.exit).resolves.not.toBe(0);
    await expect(sidecar.stop()).resolves.toBeUndefined();
  });
});
