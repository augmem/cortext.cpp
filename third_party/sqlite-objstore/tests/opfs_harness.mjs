const worker = new Worker(new URL('./opfs_worker.js', import.meta.url), {
    type: 'module'
});

function sendRoundTrip() {
    const payload = crypto.getRandomValues(new Uint8Array(16));
    worker.postMessage({
        type: 'roundtrip',
        payload: {
            bytes: payload.buffer
        }
    }, [payload.buffer]);
}

worker.addEventListener('message', (event) => {
    const { type, message, id, ok } = event.data || {};
    switch (type) {
        case 'ready':
            console.log('[opfs] worker ready, starting roundtrip test');
            sendRoundTrip();
            break;
        case 'roundtrip':
            console.log('[opfs] roundtrip complete', { ok, id });
            break;
        case 'stderr':
            console.warn('[opfs] worker stderr:', message);
            break;
        case 'error':
            console.error('[opfs] worker error:', message);
            break;
        default:
            console.log('[opfs] message', event.data);
    }
});

worker.addEventListener('error', (err) => {
    console.error('[opfs] worker crashed', err);
});

worker.postMessage({
    type: 'init',
    payload: {
        extensionPath: './objstore.wasm'
    }
});

