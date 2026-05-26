const gatewayBaseUrl = process.env.DEER_FLOW_INTERNAL_GATEWAY_BASE_URL;
const headerName = process.env.DEER_FLOW_INTERNAL_GATEWAY_HEADER_NAME;
const headerValue = process.env.DEER_FLOW_INTERNAL_GATEWAY_HEADER_VALUE;

if (gatewayBaseUrl && headerName && headerValue) {
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = requestUrl(input);
    if (!isInternalGatewayUrl(url, gatewayBaseUrl)) {
      return originalFetch(input, init);
    }

    const headers = new Headers(init?.headers ?? (input instanceof Request ? input.headers : undefined));
    headers.set(headerName, headerValue);

    return originalFetch(input, {
      ...init,
      headers,
    });
  };
}

function requestUrl(input: RequestInfo | URL): string {
  if (input instanceof Request) {
    return input.url;
  }

  return String(input);
}

function isInternalGatewayUrl(url: string, baseUrl: string): boolean {
  try {
    const target = new URL(url);
    const base = new URL(baseUrl);
    if (target.origin !== base.origin) {
      return false;
    }

    const basePath = stripTrailingSlash(base.pathname);
    return target.pathname === basePath || target.pathname.startsWith(`${basePath}/`);
  } catch {
    return false;
  }
}

function stripTrailingSlash(pathname: string): string {
  return pathname.length > 1 ? pathname.replace(/\/+$/, "") : pathname;
}
