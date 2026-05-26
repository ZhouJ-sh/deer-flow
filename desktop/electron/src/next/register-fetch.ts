const gatewayBaseUrl = process.env.DEER_FLOW_INTERNAL_GATEWAY_BASE_URL;
const headerName = process.env.DEER_FLOW_INTERNAL_GATEWAY_HEADER_NAME;
const headerValue = process.env.DEER_FLOW_INTERNAL_GATEWAY_HEADER_VALUE;

if (gatewayBaseUrl && headerName && headerValue) {
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = requestUrl(input);
    if (!url.startsWith(gatewayBaseUrl)) {
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
