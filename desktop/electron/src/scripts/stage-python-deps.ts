import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawn } from "node:child_process";

const usage = "Usage: node dist/scripts/stage-python-deps.js <backend-dir> <out-site-packages-dir>";

export type StagePythonDepsArgs = {
  backendDir: string;
  outDir: string;
};

export function parseStagePythonDepsArgs(args: string[] = process.argv.slice(2)): StagePythonDepsArgs {
  const [backendDirArg, outDirArg] = args;
  if (!outDirArg) {
    throw new Error(usage);
  }

  return {
    backendDir: resolve(backendDirArg ?? join(process.cwd(), "..", "..", "backend")),
    outDir: resolve(outDirArg),
  };
}

export async function stagePythonDeps(backendDir: string, outDir: string): Promise<void> {
  const backend = resolve(backendDir);
  const output = resolve(outDir);
  const tempDir = await mkdtemp(join(tmpdir(), "deer-flow-python-deps-"));
  const requirementsPath = join(tempDir, "requirements.txt");
  const python = process.env.DEER_FLOW_DESKTOP_BUILD_PYTHON ?? "python";

  try {
    await run("uv", [
      "export",
      "--project",
      backend,
      "--format",
      "requirements-txt",
      "--no-hashes",
      "--output-file",
      requirementsPath,
    ], { cwd: backend });
    await rm(output, { recursive: true, force: true });
    await run("uv", [
      "pip",
      "install",
      "--python",
      python,
      "--target",
      output,
      "--requirement",
      requirementsPath,
    ], { cwd: backend });
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}

async function run(command: string, args: string[], options: { cwd?: string } = {}): Promise<void> {
  await new Promise<void>((resolvePromise, reject) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      stdio: "inherit",
      env: process.env,
    });

    child.on("error", reject);
    child.on("exit", (code, signal) => {
      if (code === 0) {
        resolvePromise();
        return;
      }

      reject(new Error(`${command} ${args.join(" ")} failed with ${signal ? `signal ${signal}` : `exit code ${code}`}`));
    });
  });
}

async function main() {
  const { backendDir, outDir } = parseStagePythonDepsArgs();
  await stagePythonDeps(backendDir, outDir);
  console.log(`Python dependencies staged at ${outDir}`);
}

if (process.argv[1]?.endsWith("stage-python-deps.js")) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
