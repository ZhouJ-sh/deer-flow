import { constants } from "node:fs";
import { access, cp, mkdir, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export type RuntimeLayout = {
  backend: string;
  frontend: string;
  pythonRuntime: string;
  nodeRuntime: string;
  desktopServer: string;
};

const requiredRuntimeEnv = {
  DEER_FLOW_DESKTOP_NODE_RUNTIME_DIR: "prebuilt Node.js runtime directory",
  DEER_FLOW_DESKTOP_PYTHON_RUNTIME_DIR: "prebuilt Python runtime directory",
  DEER_FLOW_DESKTOP_PYTHON_SITE_PACKAGES_DIR: "prebuilt Python site-packages directory",
} as const;

export function runtimeLayout(_repoRoot: string, outDir: string): RuntimeLayout {
  const root = resolve(outDir);

  return {
    backend: join(root, "backend"),
    frontend: join(root, "frontend"),
    pythonRuntime: join(root, "runtimes", "python"),
    nodeRuntime: join(root, "runtimes", "node"),
    desktopServer: join(root, "desktop-server"),
  };
}

export async function stageRuntime(repoRoot: string, outDir: string): Promise<RuntimeLayout> {
  const root = resolve(repoRoot);
  const output = resolve(outDir);
  const layout = runtimeLayout(root, output);
  const env = requireRuntimeEnv();

  await assertSourceDir(env.DEER_FLOW_DESKTOP_NODE_RUNTIME_DIR, "DEER_FLOW_DESKTOP_NODE_RUNTIME_DIR");
  await assertSourceDir(env.DEER_FLOW_DESKTOP_PYTHON_RUNTIME_DIR, "DEER_FLOW_DESKTOP_PYTHON_RUNTIME_DIR");
  await assertSourceDir(env.DEER_FLOW_DESKTOP_PYTHON_SITE_PACKAGES_DIR, "DEER_FLOW_DESKTOP_PYTHON_SITE_PACKAGES_DIR");

  await rm(output, { recursive: true, force: true });
  await mkdir(output, { recursive: true });

  await Promise.all([
    copyPath(join(root, "backend", "app"), join(layout.backend, "app")),
    copyPath(join(root, "backend", "packages"), join(layout.backend, "packages")),
    copyPath(join(root, "backend", "pyproject.toml"), join(layout.backend, "pyproject.toml")),
    copyPath(join(root, "backend", "uv.lock"), join(layout.backend, "uv.lock")),
    copyPath(join(root, "config.example.yaml"), join(layout.backend, "config.example.yaml")),
    copyPath(join(root, "extensions_config.example.json"), join(layout.backend, "extensions_config.example.json")),
    copyPath(env.DEER_FLOW_DESKTOP_PYTHON_SITE_PACKAGES_DIR, join(layout.backend, "site-packages")),
    copyPath(join(root, "frontend", ".next", "standalone"), join(layout.frontend, ".next", "standalone")),
    copyPath(
      join(root, "frontend", ".next", "static"),
      join(layout.frontend, ".next", "standalone", ".next", "static"),
    ),
    copyPath(join(root, "frontend", "public"), join(layout.frontend, ".next", "standalone", "public")),
    copyPath(join(root, "desktop", "electron", "dist", "proxy"), join(layout.desktopServer, "proxy")),
    copyPath(join(root, "desktop", "electron", "dist", "next"), join(layout.desktopServer, "next")),
    copyPath(env.DEER_FLOW_DESKTOP_NODE_RUNTIME_DIR, layout.nodeRuntime),
    copyPath(env.DEER_FLOW_DESKTOP_PYTHON_RUNTIME_DIR, layout.pythonRuntime),
  ]);

  return layout;
}

function requireRuntimeEnv(): Record<keyof typeof requiredRuntimeEnv, string> {
  const values = {} as Record<keyof typeof requiredRuntimeEnv, string>;

  for (const [name, description] of Object.entries(requiredRuntimeEnv)) {
    const value = process.env[name]?.trim();
    if (!value) {
      throw new Error(
        `${name} is required. Set it to the ${description}; desktop packaging does not download runtimes or install customer-machine dependencies.`,
      );
    }
    values[name as keyof typeof requiredRuntimeEnv] = resolve(value);
  }

  return values;
}

async function copyPath(source: string, destination: string): Promise<void> {
  await cp(source, destination, {
    recursive: true,
    dereference: false,
    force: true,
    errorOnExist: false,
  });
}

async function assertSourceDir(path: string, envName: string): Promise<void> {
  try {
    await access(path, constants.R_OK);
  } catch (error) {
    throw new Error(`${envName} points to ${path}, but it is not readable.`, { cause: error });
  }
}

async function main() {
  const [repoRootArg, outDirArg] = process.argv.slice(2);
  const scriptDir = dirname(fileURLToPath(import.meta.url));
  const repoRoot = repoRootArg ? resolve(repoRootArg) : resolve(scriptDir, "..", "..", "..", "..");
  const outDir = outDirArg ? resolve(outDirArg) : join(repoRoot, "desktop", "electron", "resources");
  const layout = await stageRuntime(repoRoot, outDir);

  console.log(`Desktop runtime staged at ${outDir}`);
  console.log(JSON.stringify(layout, null, 2));
}

if (process.argv[1]?.endsWith("build-runtime.js")) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
