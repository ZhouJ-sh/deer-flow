import { join, resolve } from "node:path";

export interface ResolveDesktopResourcesOptions {
  packaged: boolean;
  appPath: string;
  resourcesPath: string;
  platform?: NodeJS.Platform;
}

export interface DesktopResources {
  repoRoot: string | null;
  backendDir: string;
  frontendDir: string;
  pythonBin: string;
  nodeBin: string;
  desktopServerDir: string;
}

export function resolveDesktopResources(options: ResolveDesktopResourcesOptions): DesktopResources {
  const appPath = resolve(options.appPath);
  const resourcesPath = resolve(options.resourcesPath);
  const platform = options.platform ?? process.platform;

  if (!options.packaged) {
    const repoRoot = resolve(appPath, "..", "..");

    return {
      repoRoot,
      backendDir: join(repoRoot, "backend"),
      frontendDir: join(repoRoot, "frontend"),
      pythonBin: "python",
      nodeBin: "node",
      desktopServerDir: join(appPath, "dist"),
    };
  }

  const pythonBin =
    platform === "win32"
      ? join(resourcesPath, "runtimes", "python", "python.exe")
      : join(resourcesPath, "runtimes", "python", "bin", "python");
  const nodeBin =
    platform === "win32"
      ? join(resourcesPath, "runtimes", "node", "node.exe")
      : join(resourcesPath, "runtimes", "node", "bin", "node");

  return {
    repoRoot: null,
    backendDir: join(resourcesPath, "backend"),
    frontendDir: join(resourcesPath, "frontend"),
    pythonBin,
    nodeBin,
    desktopServerDir: join(resourcesPath, "desktop-server"),
  };
}
