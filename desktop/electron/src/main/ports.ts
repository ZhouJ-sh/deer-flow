import getPort from "get-port";

export interface DesktopPorts {
  gatewayPort: number;
  nextPort: number;
  proxyPort: number;
}

export async function allocateDesktopPorts(): Promise<DesktopPorts> {
  const allocated = new Set<number>();
  const getLoopbackPort = async () => {
    const port = await getPort({
      host: "127.0.0.1",
      exclude: Array.from(allocated),
    });
    allocated.add(port);
    return port;
  };

  return {
    gatewayPort: await getLoopbackPort(),
    nextPort: await getLoopbackPort(),
    proxyPort: await getLoopbackPort(),
  };
}
