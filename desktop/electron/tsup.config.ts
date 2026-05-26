import { defineConfig, type Options } from "tsup";

const sharedConfig = {
  platform: "node",
  target: "node20",
  outDir: "dist",
  splitting: false,
  sourcemap: true,
  external: ["electron"],
} satisfies Options;

export default defineConfig([
  {
    ...sharedConfig,
    entry: {
      "main/index": "src/main/index.ts",
      "proxy/desktop-proxy": "src/proxy/desktop-proxy.ts",
    },
    format: ["esm"],
    clean: true,
  },
  {
    ...sharedConfig,
    entry: {
      "preload/index": "src/preload/index.ts",
    },
    format: ["cjs"],
    outExtension: () => ({ js: ".js" }),
  },
]);
