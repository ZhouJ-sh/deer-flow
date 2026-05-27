import { createServer } from "node:net";

import { describe, expect, test } from "vitest";

import { allocateDesktopPorts } from "../src/main/ports.js";

async function assertBindable(port: number) {
  const server = createServer();

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });

  await new Promise<void>((resolve, reject) => {
    server.close((error) => {
      if (error) {
        reject(error);
      } else {
        resolve();
      }
    });
  });
}

describe("allocateDesktopPorts", () => {
  test("allocates three distinct ports bindable on 127.0.0.1", async () => {
    const ports = await allocateDesktopPorts();
    const values = [ports.gatewayPort, ports.nextPort, ports.proxyPort];

    expect(new Set(values).size).toBe(3);
    for (const port of values) {
      expect(port).toBeGreaterThan(0);
      await assertBindable(port);
    }
  });
});
