import { createWriteStream } from "node:fs";
import { ChildProcess, spawn } from "node:child_process";
import type { WriteStream } from "node:fs";

export interface StartSidecarOptions {
  name: string;
  command: string;
  args: string[];
  cwd: string;
  env: NodeJS.ProcessEnv;
  logPath: string;
}

export interface SidecarProcess {
  name: string;
  child: ChildProcess;
  exit: Promise<number | null>;
  stop: () => Promise<void>;
}

const STOP_TIMEOUT_MS = 5_000;

export function startSidecar(options: StartSidecarOptions): SidecarProcess {
  const log = createWriteStream(options.logPath, { flags: "a" });
  const logLifecycle = trackLogLifecycle(log);
  const child = spawn(options.command, options.args, {
    cwd: options.cwd,
    env: options.env,
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });

  child.stdout?.pipe(log, { end: false });
  child.stderr?.pipe(log, { end: false });

  let closed = false;
  let spawnError: Error | null = null;
  const childClosed = new Promise<number | null>((resolve) => {
    child.once("error", (error) => {
      spawnError = error;
    });

    child.once("close", (code) => {
      closed = true;
      resolve(code);
    });
  });

  const exit = (async () => {
    const code = await childClosed;
    await logLifecycle.end();
    if (spawnError) {
      throw spawnError;
    }
    if (logLifecycle.error) {
      throw logLifecycle.error;
    }
    return code;
  })();

  const stop = async () => {
    if (closed || child.exitCode !== null || child.signalCode !== null) {
      return;
    }

    child.kill(process.platform === "win32" ? undefined : "SIGTERM");

    const timeout = new Promise<"timeout">((resolve) => {
      setTimeout(() => resolve("timeout"), STOP_TIMEOUT_MS).unref();
    });

    if ((await Promise.race([childClosed, timeout])) === "timeout" && !closed) {
      child.kill("SIGKILL");
      await childClosed;
    }
  };

  return {
    name: options.name,
    child,
    exit,
    stop,
  };
}

function trackLogLifecycle(log: WriteStream) {
  let error: Error | null = null;
  let complete = false;
  const done = new Promise<void>((resolve) => {
    const markComplete = () => {
      complete = true;
      resolve();
    };
    log.once("finish", markComplete);
    log.once("close", markComplete);
  });

  log.once("error", (err) => {
    error = err;
  });

  return {
    get error() {
      return error;
    },
    async end() {
      if (!complete && !log.destroyed && !log.writableEnded) {
        log.end();
      }
      await done;
    },
  };
}
