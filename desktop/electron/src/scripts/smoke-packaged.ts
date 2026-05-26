import { constants } from "node:fs";
import { access } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { runtimeLayout } from "./build-runtime.js";

export async function smokePackaged(resourcesPath: string): Promise<void> {
  const root = resolve(resourcesPath);
  const layout = runtimeLayout(root, root);
  const requiredPaths = [
    layout.backend,
    join(layout.backend, "app"),
    join(layout.backend, "packages"),
    join(layout.backend, "pyproject.toml"),
    join(layout.backend, "uv.lock"),
    join(layout.backend, "config.example.yaml"),
    join(layout.backend, "extensions_config.example.json"),
    join(layout.backend, "site-packages"),
    join(layout.frontend, ".next", "standalone"),
    join(layout.frontend, ".next", "standalone", ".next", "static"),
    join(layout.frontend, ".next", "standalone", "public"),
    join(layout.desktopServer, "proxy"),
    join(layout.desktopServer, "next"),
    layout.nodeRuntime,
    layout.pythonRuntime,
  ];

  const missing: string[] = [];
  for (const path of requiredPaths) {
    if (!(await exists(path))) {
      missing.push(path);
    }
  }

  if (missing.length > 0) {
    throw new Error(`Packaged runtime layout smoke failed. Missing required resources:\n${missing.join("\n")}`);
  }

  console.log("Packaged runtime layout smoke passed");
}

async function exists(path: string): Promise<boolean> {
  try {
    await access(path, constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

async function main() {
  const [resourcesPathArg] = process.argv.slice(2);
  const scriptDir = dirname(fileURLToPath(import.meta.url));
  const resourcesPath =
    resourcesPathArg ?? resolve(scriptDir, "..", "..", "resources");

  await smokePackaged(resourcesPath);
}

if (process.argv[1]?.endsWith("smoke-packaged.js")) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
