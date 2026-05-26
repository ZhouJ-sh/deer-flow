import { randomBytes, randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { access, chmod, copyFile, cp, mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";

import { parse, stringify } from "yaml";

export type DesktopDataOptions = {
  root: string;
  exampleConfigPath?: string;
  exampleExtensionsConfigPath?: string;
  bundledSkillsPath?: string;
};

export type DesktopDataPaths = {
  root: string;
  deerFlowHome: string;
  dataDir: string;
  logsDir: string;
  runtimeDir: string;
  sqliteDir: string;
  configPath: string;
  extensionsConfigPath: string;
  envPath: string;
  tokenPath: string;
  installIdPath: string;
  skillsPath: string;
  exampleConfigPath?: string;
  exampleExtensionsConfigPath?: string;
  bundledSkillsPath?: string;
};

const userOnlyFileMode = 0o600;

export async function ensureDesktopData(options: DesktopDataOptions): Promise<DesktopDataPaths> {
  const root = resolve(options.root);
  const deerFlowHome = join(root, ".deer-flow");
  const dataDir = join(deerFlowHome, "data");
  const paths: DesktopDataPaths = {
    root,
    deerFlowHome,
    dataDir,
    logsDir: join(root, "logs"),
    runtimeDir: join(root, "runtime"),
    sqliteDir: dataDir,
    configPath: join(root, "config.yaml"),
    extensionsConfigPath: join(root, "extensions_config.json"),
    envPath: join(root, ".env"),
    tokenPath: join(root, "desktop-token"),
    installIdPath: join(root, "install-id"),
    skillsPath: join(root, "skills"),
    exampleConfigPath: options.exampleConfigPath,
    exampleExtensionsConfigPath: options.exampleExtensionsConfigPath,
    bundledSkillsPath: options.bundledSkillsPath,
  };

  await Promise.all([
    mkdir(paths.deerFlowHome, { recursive: true }),
    mkdir(paths.dataDir, { recursive: true }),
    mkdir(paths.logsDir, { recursive: true }),
    mkdir(paths.runtimeDir, { recursive: true }),
  ]);

  await ensureConfig(paths);
  await ensureExtensionsConfig(paths);
  await ensureTextFile(paths.envPath, "");
  await ensureSecretFile(paths.tokenPath, `${randomSecret()}\n`);
  await ensureSecretFile(paths.installIdPath, `${randomUUID()}\n`);
  await ensureBundledSkills(paths);

  return paths;
}

async function ensureBundledSkills(paths: DesktopDataPaths): Promise<void> {
  if (await exists(paths.skillsPath)) {
    return;
  }

  if (paths.bundledSkillsPath && (await exists(paths.bundledSkillsPath))) {
    await cp(paths.bundledSkillsPath, paths.skillsPath, {
      recursive: true,
      dereference: false,
      force: false,
      errorOnExist: true,
    });
    return;
  }

  await mkdir(paths.skillsPath, { recursive: true });
}

export async function ensureSecretTextFile(path: string, createValue: () => string): Promise<string> {
  let current = "";
  try {
    current = await readFile(path, "utf8");
  } catch (error) {
    if (!isNotFoundError(error)) {
      throw error;
    }
  }

  if (current.trim().length > 0) {
    await chmodUserOnly(path);
    return current;
  }

  const value = createValue();
  await writeFile(path, value, { mode: userOnlyFileMode });
  await chmodUserOnly(path);
  return value;
}

function randomSecret(): string {
  return randomBytes(48).toString("base64url");
}

async function ensureConfig(paths: DesktopDataPaths): Promise<void> {
  await ensureTextFile(paths.configPath, fallbackConfig(), paths.exampleConfigPath);

  const rawConfig = await readFile(paths.configPath, "utf8");
  const parsed = parse(rawConfig);
  const config = isRecord(parsed) ? parsed : {};

  delete config.checkpointer;
  config.database = {
    backend: "sqlite",
    sqlite_dir: paths.sqliteDir,
  };
  config.run_events = {
    backend: "db",
  };
  config.sandbox = {
    use: "deerflow.sandbox.local:LocalSandboxProvider",
    allow_host_bash: false,
  };

  await writeFile(paths.configPath, stringify(config), { mode: userOnlyFileMode });
  await chmodUserOnly(paths.configPath);
}

async function ensureExtensionsConfig(paths: DesktopDataPaths): Promise<void> {
  if (await exists(paths.extensionsConfigPath)) {
    await chmodUserOnly(paths.extensionsConfigPath);
    return;
  }

  if (paths.exampleExtensionsConfigPath && (await exists(paths.exampleExtensionsConfigPath))) {
    await copyFile(paths.exampleExtensionsConfigPath, paths.extensionsConfigPath);
    await chmodUserOnly(paths.extensionsConfigPath);
    return;
  }

  await ensureTextFile(paths.extensionsConfigPath, `${JSON.stringify({ mcpServers: {}, skills: {} }, null, 2)}\n`);
}

async function ensureTextFile(path: string, fallbackContent: string, sourcePath?: string): Promise<void> {
  if (await exists(path)) {
    await chmodUserOnly(path);
    return;
  }

  if (sourcePath && (await exists(sourcePath))) {
    await copyFile(sourcePath, path);
    await chmodUserOnly(path);
    return;
  }

  await writeFile(path, fallbackContent, { mode: userOnlyFileMode });
  await chmodUserOnly(path);
}

async function ensureSecretFile(path: string, value: string): Promise<void> {
  await ensureSecretTextFile(path, () => value);
}

async function exists(path: string): Promise<boolean> {
  try {
    await access(path, constants.F_OK);
    return true;
  } catch (error) {
    if (isNotFoundError(error)) {
      return false;
    }
    throw error;
  }
}

async function chmodUserOnly(path: string): Promise<void> {
  if (process.platform === "win32") {
    await chmodUserOnlyBestEffort(path);
    return;
  }

  const pathStat = await stat(path);
  if (pathStat.isFile()) {
    await chmod(path, userOnlyFileMode);
  }
}

async function chmodUserOnlyBestEffort(path: string): Promise<void> {
  try {
    const pathStat = await stat(path);
    if (pathStat.isFile()) {
      await chmod(path, userOnlyFileMode);
    }
  } catch (error) {
    return;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNotFoundError(error: unknown): boolean {
  return isNodeError(error) && error.code === "ENOENT";
}

function isNodeError(error: unknown): error is NodeJS.ErrnoException {
  return error instanceof Error && "code" in error;
}

function fallbackConfig(): string {
  return stringify({
    log_level: "info",
    database: {
      backend: "sqlite",
    },
    run_events: {
      backend: "db",
    },
    sandbox: {
      use: "deerflow.sandbox.local:LocalSandboxProvider",
      allow_host_bash: false,
    },
  });
}
