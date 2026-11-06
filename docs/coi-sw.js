// GPU가 없는 기기에서만 등록됨: 페이지에 COOP/COEP 헤더를 붙여 crossOriginIsolated를 켜고,
// CPU 모델(wllama)이 여러 스레드를 쓸 수 있게 함. GitHub Pages는 헤더를 직접 못 바꿔서 서비스 워커로 붙임.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.mode !== 'navigate') return;
  e.respondWith(fetch(req).then(res => {
    if (res.status === 0 || res.type === 'opaqueredirect') return res;
    const headers = new Headers(res.headers);
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    headers.set('Cross-Origin-Embedder-Policy', 'credentialless');
    return new Response(res.body, {status: res.status, statusText: res.statusText, headers});
  }).catch(() => fetch(req)));
});
