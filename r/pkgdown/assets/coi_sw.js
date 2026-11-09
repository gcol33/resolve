// Cross-origin isolation for the browser app (articles/app.html). Workers can
// share memory (SharedArrayBuffer) only on a cross-origin isolated page, which
// takes two response headers the site's host does not send. This service
// worker adds them to the app page and to the workers it starts, and passes
// every other request through untouched. `credentialless` lets the page keep
// loading cross-origin scripts, styles and map tiles that carry no
// Cross-Origin-Resource-Policy header; a browser that does not know it treats
// the page as not isolated, and the app runs its searches on one thread.

const APP_PAGE = /\/articles\/app\.html$/;

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

function isolates(request) {
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return false;
  if (request.mode === "navigate") return APP_PAGE.test(url.pathname);
  return request.destination === "worker";
}

async function withIsolation(request) {
  const response = await fetch(request);
  if (response.status === 0) return response;
  const headers = new Headers(response.headers);
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  headers.set("Cross-Origin-Embedder-Policy", "credentialless");
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

self.addEventListener("fetch", (event) => {
  if (isolates(event.request)) event.respondWith(withIsolation(event.request));
});
