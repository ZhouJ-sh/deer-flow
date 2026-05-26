import { contextBridge } from "electron";

contextBridge.exposeInMainWorld("deerFlowDesktop", {
  version: "0.1.0",
});
