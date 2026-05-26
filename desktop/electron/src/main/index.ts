import { app, BrowserWindow } from "electron";

async function main() {
  await app.whenReady();
  const win = new BrowserWindow({
    width: 1280,
    height: 840,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  await win.loadURL("about:blank");
}

void main();
