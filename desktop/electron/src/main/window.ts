import electron from "electron";
import type { BrowserWindow as BrowserWindowInstance } from "electron";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const { BrowserWindow, shell } = electron;
const __dirname = dirname(fileURLToPath(import.meta.url));

export async function createDesktopWindow(url: string): Promise<BrowserWindowInstance> {
  const window = new BrowserWindow({
    width: 1280,
    height: 840,
    minWidth: 960,
    minHeight: 640,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      preload: join(__dirname, "..", "preload", "index.js"),
    },
  });

  const appOrigin = new URL(url).origin;

  window.webContents.setWindowOpenHandler(({ url: targetUrl }) => {
    if (shouldOpenExternalUrl(targetUrl)) {
      void shell.openExternal(targetUrl);
    }
    return { action: "deny" };
  });

  window.webContents.on("will-navigate", (event, targetUrl) => {
    if (new URL(targetUrl).origin === appOrigin) {
      return;
    }

    event.preventDefault();
    if (shouldOpenExternalUrl(targetUrl)) {
      void shell.openExternal(targetUrl);
    }
  });

  await window.loadURL(url);
  return window;
}

export function shouldOpenExternalUrl(targetUrl: string): boolean {
  try {
    const protocol = new URL(targetUrl).protocol;
    return protocol === "http:" || protocol === "https:" || protocol === "mailto:";
  } catch {
    return false;
  }
}
