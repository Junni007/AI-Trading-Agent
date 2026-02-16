/**
 * Service Worker for Signal.Engine PWA
 * Provides offline caching and improved performance
 */

const CACHE_NAME = 'signal-engine-v1';
const STATIC_ASSETS = [
    '/',
    '/index.html',
    '/manifest.json',
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            console.log('[SW] Caching static assets');
            return cache.addAll(STATIC_ASSETS);
        })
    );
    self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames
                    .filter((name) => name !== CACHE_NAME)
                    .map((name) => {
                        console.log('[SW] Deleting old cache:', name);
                        return caches.delete(name);
                    })
            );
        })
    );
    self.clients.claim();
});

// Fetch event - network-first strategy for API, cache-first for static
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);

    // Skip non-GET requests
    if (request.method !== 'GET') return;

    // Skip WebSocket and API requests (always go to network)
    if (url.pathname.startsWith('/api') || url.pathname.startsWith('/ws')) {
        return;
    }

    // For static assets, use cache-first strategy
    event.respondWith(
        caches.match(request).then((cachedResponse) => {
            if (cachedResponse) {
                // Return cached version but update cache in background
                event.waitUntil(
                    fetch(request)
                        .then((response) => {
                            if (response.ok) {
                                caches.open(CACHE_NAME).then((cache) => {
                                    cache.put(request, response);
                                });
                            }
                        })
                        .catch(() => { })
                );
                return cachedResponse;
            }

            // Not in cache, fetch from network
            return fetch(request)
                .then((response) => {
                    // Cache successful responses
                    if (response.ok && response.type === 'basic') {
                        const responseClone = response.clone();
                        caches.open(CACHE_NAME).then((cache) => {
                            cache.put(request, responseClone);
                        });
                    }
                    return response;
                })
                .catch(() => {
                    // Return offline fallback if available
                    if (request.destination === 'document') {
                        return caches.match('/');
                    }
                    return new Response('Offline', { status: 503 });
                });
        })
    );
});

// Listen for messages from main thread
self.addEventListener('message', (event) => {
    if (event.data === 'skipWaiting') {
        self.skipWaiting();
    }
});
