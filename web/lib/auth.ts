export const API_TOKEN_STORAGE_KEY = "kb_arena_api_token";

export function getApiToken(): string {
  if (typeof window === "undefined") return "";
  try {
    return window.sessionStorage.getItem(API_TOKEN_STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setApiToken(token: string): void {
  if (typeof window === "undefined") return;
  try {
    const value = token.trim();
    if (value) window.sessionStorage.setItem(API_TOKEN_STORAGE_KEY, value);
    else window.sessionStorage.removeItem(API_TOKEN_STORAGE_KEY);
  } catch {
    // Storage can be unavailable in hardened browser contexts.
  }
}

export function clearApiToken(): void {
  setApiToken("");
}

export function apiFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = getApiToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return fetch(input, { ...init, headers });
}
