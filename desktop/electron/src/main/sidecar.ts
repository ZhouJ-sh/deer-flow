import { createWriteStream } from "node:fs";
import { ChildProcess, spawn } from "node:child_process";

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
  const child = spawn(options.command, options.args, {
    cwd: options.cwd,
    env: options.env,
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });

  child.stdout?.pipe(log, { end: false });
  child.stderr?.pipe(log, { end: false });

  let exited = false;
  const exit = new Promise<number | null>((resolve, reject) => {
    child.once("error", (error) => {
      exited = true;
      log.end();
      reject(error);
    });

    child.once("exit", (code) => {
      exited = true;
      log.end();
      resolve(code);
    });
  });

  const stop = async () => {
    if (exited || child.exitCode !== null || child.signalCode !== null) {
      return;
    }

    const settled = new Promise<void>((resolve) => {
      child.once("exit", () => resolve());
      child.once("error", () => resolve());
    });

    child.kill(process.platform === "win32" ? undefined : "SIGTERM");

    const timeout = new Promise<"timeout">((resolve) => {
      setTimeout(() => resolve("timeout"), STOP_TIMEOUT_MS).unref();
    });

    if ((await Promise.race([settled, timeout])) === "timeout" && !exited) {
      child.kill("SIGKILL");
      await settled;
    }
  };

  return {
    name: options.name,
    child,
    exit,
    stop,
  };
}
