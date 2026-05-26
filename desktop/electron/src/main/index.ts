import { app } from "electron";

import { startDesktopRuntime, type DesktopRuntime } from "./runtime.js";
import { createDesktopWindow } from "./window.js";

async function main() {
  await app.whenReady();

  const smokeUrl = process.env.DEER_FLOW_DESKTOP_SMOKE_URL;
  if (smokeUrl) {
    await createDesktopWindow(smokeUrl);
    return;
  }

  const runtime = await startDesktopRuntime({
    appDataRoot: app.getPath("userData"),
    packaged: app.isPackaged,
    appPath: app.getAppPath(),
    resourcesPath: process.resourcesPath,
  });
  registerRuntimeShutdown(runtime);

  await createDesktopWindow(runtime.proxyOrigin);
}

function registerRuntimeShutdown(runtime: DesktopRuntime) {
  let stopping = false;

  app.on("before-quit", (event) => {
    if (stopping) {
      return;
    }

    event.preventDefault();
    stopping = true;
    void runtime.stop().finally(() => app.quit());
  });
}

main().catch((error) => {
  console.error(error);
  app.exit(1);
});
