import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, test } from "vitest";

import { startSidecar } from "../src/main/sidecar.js";

const originalEnv = { ...process.env };
let created: string[] = [];

afterEach(async () => {
  process.env = { ...originalEnv };
  await Promise.all(created.map((dir) => rm(dir, { recursive: true, force: true })));
  created = [];
});

async function tempLogPath(name: string) {
  return join(await tempDir(), name);
}

async function tempDir() {
  const dir = await mkdtemp(join(tmpdir(), "deer-flow-sidecar-"));
  created.push(dir);
  return dir;
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

  test("surfaces log file errors through the exit promise", async () => {
    const logPath = join(await tempDir(), "missing", "output.log");
    const sidecar = startSidecar({
      name: "bad-log",
      command: process.execPath,
      args: ["-e", "console.log('cannot be logged')"],
      cwd: process.cwd(),
      env: {},
      logPath,
    });

    await expect(sidecar.exit).rejects.toMatchObject({ code: "ENOENT" });
  });

  test("uses the provided environment without inheriting parent env vars", async () => {
    process.env.DEER_FLOW_PARENT_ONLY = "should-not-leak";
    const logPath = await tempLogPath("env.log");
    const sidecar = startSidecar({
      name: "env-writer",
      command: process.execPath,
      args: [
        "-e",
        [
          "console.log(JSON.stringify({",
          "provided: process.env.DEER_FLOW_PROVIDED_ONLY,",
          "parent: process.env.DEER_FLOW_PARENT_ONLY ?? null",
          "}));",
        ].join(""),
      ],
      cwd: process.cwd(),
      env: {
        PATH: process.env.PATH,
        DEER_FLOW_PROVIDED_ONLY: "kept",
      },
      logPath,
    });

    await expect(sidecar.exit).resolves.toBe(0);
    await expect(readFile(logPath, "utf8")).resolves.toContain('{"provided":"kept","parent":null}');
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
