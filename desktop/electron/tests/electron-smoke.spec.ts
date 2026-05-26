import { test, expect, _electron as electron } from "@playwright/test";
import { join } from "node:path";

import { startStubFrontend } from "./fixtures/stub-runtime.js";

test("opens a local shell window with a stub runtime", async () => {
  const stub = await startStubFrontend();
  let app: Awaited<ReturnType<typeof electron.launch>> | null = null;

  try {
    app = await electron.launch({
      args: [join(process.cwd(), "dist", "main", "index.js")],
      env: {
        ...process.env,
        DEER_FLOW_DESKTOP_SMOKE_URL: stub.url,
      },
    });

    const page = await app.firstWindow();
    await expect(page.locator("main")).toContainText("DeerFlow Desktop Smoke");
  } finally {
    try {
      await app?.close();
    } finally {
      await stub.close();
    }
  }
});
