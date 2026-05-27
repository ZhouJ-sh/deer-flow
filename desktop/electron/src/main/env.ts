import { randomBytes } from "node:crypto";
import { join } from "node:path";

import { configDotenv } from "dotenv";

import { type DesktopDataPaths, ensureSecretTextFile } from "./app-data.js";

type DesktopEnv = NodeJS.ProcessEnv;

export async function loadDesktopDotEnv(envPath: string): Promise<Record<string, string>> {
  const parsed = configDotenv({ path: envPath, processEnv: {} }).parsed;
  return parsed ?? {};
}

export async function buildGatewayEnv(
  paths: DesktopDataPaths,
  options: { frontendOrigin: string },
): Promise<DesktopEnv> {
  return {
    ...process.env,
    ...(await loadDesktopDotEnv(paths.envPath)),
    DEER_FLOW_DESKTOP: "1",
    DEER_FLOW_HOME: paths.deerFlowHome,
    DEER_FLOW_CONFIG_PATH: paths.configPath,
    DEER_FLOW_EXTENSIONS_CONFIG_PATH: paths.extensionsConfigPath,
    DEER_FLOW_PROJECT_ROOT: paths.root,
    DEER_FLOW_SKILLS_PATH: paths.skillsPath,
    DEER_FLOW_DESKTOP_TOKEN_FILE: paths.tokenPath,
    GATEWAY_HOST: "127.0.0.1",
    GATEWAY_CORS_ORIGINS: options.frontendOrigin,
  };
}

export async function buildNextEnv(paths: DesktopDataPaths, options: { proxyOrigin: string }): Promise<DesktopEnv> {
  const env: DesktopEnv = {
    ...process.env,
    ...(await loadDesktopDotEnv(paths.envPath)),
    BETTER_AUTH_SECRET: await ensureBetterAuthSecret(paths),
    DEER_FLOW_INTERNAL_GATEWAY_BASE_URL: `${options.proxyOrigin}/_desktop-gateway`,
    DEER_FLOW_TRUSTED_ORIGINS: options.proxyOrigin,
    SKIP_ENV_VALIDATION: "1",
  };

  delete env.NEXT_PUBLIC_LANGGRAPH_BASE_URL;
  delete env.NEXT_PUBLIC_BACKEND_BASE_URL;

  return env;
}

async function ensureBetterAuthSecret(paths: DesktopDataPaths): Promise<string> {
  const value = await ensureSecretTextFile(join(paths.deerFlowHome, "better-auth-secret"), () => {
    return `${cryptoRandomSecret()}\n`;
  });
  return value.trim();
}

function cryptoRandomSecret(): string {
  return randomBytes(48).toString("base64url");
}
