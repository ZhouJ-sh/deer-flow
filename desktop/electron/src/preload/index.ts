import { contextBridge } from "electron";

contextBridge.exposeInMainWorld("deerFlowDesktop", {
  platform: process.platform,
});
