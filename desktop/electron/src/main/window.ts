import { BrowserWindow, shell } from "electron";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

export async function createDesktopWindow(url: string): Promise<BrowserWindow> {
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
    void shell.openExternal(targetUrl);
    return { action: "deny" };
  });

  window.webContents.on("will-navigate", (event, targetUrl) => {
    if (new URL(targetUrl).origin === appOrigin) {
      return;
    }

    event.preventDefault();
    void shell.openExternal(targetUrl);
  });

  await window.loadURL(url);
  return window;
}
