import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

export async function stagePythonDeps(repoRoot: string, outDir: string): Promise<void> {
  const root = resolve(repoRoot);
  const output = resolve(outDir);
  const tempDir = await mkdtemp(join(tmpdir(), "deer-flow-python-deps-"));
  const requirementsPath = join(tempDir, "requirements.txt");
  const python = process.env.DEER_FLOW_DESKTOP_BUILD_PYTHON ?? "python";

  try {
    await run("uv", [
      "export",
      "--project",
      join(root, "backend"),
      "--format",
      "requirements-txt",
      "--no-hashes",
      "--output-file",
      requirementsPath,
    ]);
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
    ]);
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}

async function run(command: string, args: string[]): Promise<void> {
  await new Promise<void>((resolvePromise, reject) => {
    const child = spawn(command, args, {
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
  const [outDirArg, repoRootArg] = process.argv.slice(2);
  if (!outDirArg) {
    throw new Error("Usage: node dist/scripts/stage-python-deps.js <out-dir> [repo-root]");
  }

  const scriptDir = dirname(fileURLToPath(import.meta.url));
  const repoRoot = repoRootArg ? resolve(repoRootArg) : resolve(scriptDir, "..", "..", "..", "..");
  await stagePythonDeps(repoRoot, outDirArg);
  console.log(`Python dependencies staged at ${resolve(outDirArg)}`);
}

if (process.argv[1]?.endsWith("stage-python-deps.js")) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
