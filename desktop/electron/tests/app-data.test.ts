import { mkdtemp, readFile, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { describe, expect, test } from "vitest";
import { parse } from "yaml";

import { ensureDesktopData } from "../src/main/app-data.js";

async function makeRoot() {
  return mkdtemp(join(tmpdir(), "deer-flow-desktop-app-data-"));
}

const repoRoot = resolve(import.meta.dirname, "../../..");
const exampleConfigPath = join(repoRoot, "config.example.yaml");
const exampleExtensionsConfigPath = join(repoRoot, "extensions_config.example.json");

describe("ensureDesktopData", () => {
  test("creates stable app-data paths from examples and sanitizes desktop config", async () => {
    const root = await makeRoot();

    const paths = await ensureDesktopData({
      root,
      exampleConfigPath,
      exampleExtensionsConfigPath,
    });

    expect(paths).toEqual({
      root,
      deerFlowHome: join(root, ".deer-flow"),
      dataDir: join(root, ".deer-flow", "data"),
      logsDir: join(root, "logs"),
      runtimeDir: join(root, "runtime"),
      sqliteDir: join(root, ".deer-flow", "data"),
      configPath: join(root, "config.yaml"),
      extensionsConfigPath: join(root, "extensions_config.json"),
      envPath: join(root, ".env"),
      tokenPath: join(root, "desktop-token"),
      installIdPath: join(root, "install-id"),
      exampleConfigPath,
      exampleExtensionsConfigPath,
    });

    await expect(stat(paths.deerFlowHome)).resolves.toMatchObject({ isDirectory: expect.any(Function) });
    await expect(stat(paths.dataDir)).resolves.toMatchObject({ isDirectory: expect.any(Function) });
    await expect(stat(paths.logsDir)).resolves.toMatchObject({ isDirectory: expect.any(Function) });
    await expect(stat(paths.runtimeDir)).resolves.toMatchObject({ isDirectory: expect.any(Function) });
    await expect(readFile(paths.envPath, "utf8")).resolves.toBe("");

    const config = parse(await readFile(paths.configPath, "utf8"));
    expect(config.log_level).toBe("info");
    expect(config.database).toEqual({
      backend: "sqlite",
      sqlite_dir: paths.sqliteDir,
    });
    expect(config.run_events).toEqual({ backend: "db" });
    expect(config.checkpointer).toBeUndefined();
    expect(config.sandbox).toEqual({
      use: "deerflow.sandbox.local:LocalSandboxProvider",
      allow_host_bash: false,
    });

    const extensions = JSON.parse(await readFile(paths.extensionsConfigPath, "utf8"));
    expect(extensions.mcpServers.github.enabled).toBe(false);
  });

  test("re-sanitizes stale config while preserving unrelated fields and existing extensions config", async () => {
    const root = await makeRoot();
    await ensureDesktopData({ root, exampleConfigPath, exampleExtensionsConfigPath });

    const customExtensions = { mcpServers: { local: { enabled: true } }, skills: { keep: true } };
    await writeFile(
      join(root, "config.yaml"),
      [
        "log_level: debug",
        "models:",
        "  - name: custom",
        "    use: provider:Model",
        "tools:",
        "  enabled: true",
        "database:",
        "  backend: postgres",
        "  uri: postgresql://localhost/deerflow",
        "run_events:",
        "  backend: memory",
        "checkpointer:",
        "  backend: redis",
        "sandbox:",
        "  use: deerflow.sandbox.container:ContainerSandboxProvider",
        "  allow_host_bash: true",
        "  container_prefix: stale",
        "  image: deer-flow/aio",
        "  docker_host: unix:///var/run/docker.sock",
        "  aio_endpoint: http://localhost:3000",
        "",
      ].join("\n"),
    );
    await writeFile(join(root, "extensions_config.json"), `${JSON.stringify(customExtensions, null, 2)}\n`);

    const paths = await ensureDesktopData({ root, exampleConfigPath, exampleExtensionsConfigPath });

    const config = parse(await readFile(paths.configPath, "utf8"));
    expect(config.log_level).toBe("debug");
    expect(config.models).toEqual([{ name: "custom", use: "provider:Model" }]);
    expect(config.tools).toEqual({ enabled: true });
    expect(config.database).toEqual({ backend: "sqlite", sqlite_dir: paths.sqliteDir });
    expect(config.run_events).toEqual({ backend: "db" });
    expect(config.checkpointer).toBeUndefined();
    expect(config.sandbox).toEqual({
      use: "deerflow.sandbox.local:LocalSandboxProvider",
      allow_host_bash: false,
    });
    expect(config.sandbox.container_prefix).toBeUndefined();
    expect(config.sandbox.image).toBeUndefined();
    expect(config.sandbox.docker_host).toBeUndefined();
    expect(config.sandbox.aio_endpoint).toBeUndefined();

    await expect(readFile(paths.extensionsConfigPath, "utf8")).resolves.toBe(
      `${JSON.stringify(customExtensions, null, 2)}\n`,
    );
  });

  test("persists existing token and install id, and replaces empty secrets", async () => {
    const root = await makeRoot();
    const first = await ensureDesktopData({ root });
    const token = await readFile(first.tokenPath, "utf8");
    const installId = await readFile(first.installIdPath, "utf8");

    expect(token).toMatch(/^[A-Za-z0-9_-]{43,}\n$/);
    expect(installId).toMatch(/^[0-9a-f-]{36}\n$/);

    const second = await ensureDesktopData({ root });
    await expect(readFile(second.tokenPath, "utf8")).resolves.toBe(token);
    await expect(readFile(second.installIdPath, "utf8")).resolves.toBe(installId);

    await writeFile(second.tokenPath, "");
    await writeFile(second.installIdPath, "");

    const third = await ensureDesktopData({ root });
    const newToken = await readFile(third.tokenPath, "utf8");
    const newInstallId = await readFile(third.installIdPath, "utf8");
    expect(newToken).toMatch(/^[A-Za-z0-9_-]{43,}\n$/);
    expect(newToken).not.toBe(token);
    expect(newInstallId).toMatch(/^[0-9a-f-]{36}\n$/);
    expect(newInstallId).not.toBe(installId);
  });
});
