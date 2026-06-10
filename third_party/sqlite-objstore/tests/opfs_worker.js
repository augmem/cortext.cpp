/* eslint-disable no-restricted-globals */
/* global sqlite3InitModule */

let sqlite3;
let db;

async function ensureDatabase(config) {
    if (db) {
        return;
    }
    if (typeof sqlite3InitModule !== 'function') {
        throw new Error('sqlite3InitModule is not defined. Load sqlite3.js before the worker.');
    }
    sqlite3 = await sqlite3InitModule({
        printErr: (msg) => postMessage({ type: 'stderr', message: msg }),
    });
    db = new sqlite3.oo1.DB(config?.filename || 'opfs.db');
    if (config?.extensionPath) {
        db.exec(`SELECT load_extension('${config.extensionPath}')`);
    }
    db.exec('CREATE VIRTUAL TABLE IF NOT EXISTS objstore USING objstore()');
}

async function handleRoundTrip(payload) {
    await ensureDatabase(payload);
    const bytes = payload?.bytes
        ? new Uint8Array(payload.bytes)
        : new Uint8Array([1, 2, 3, 4]);
    db.exec('BEGIN');
    const stmt = db.prepare('INSERT INTO objstore(data) VALUES (?)');
    stmt.bind([bytes]);
    stmt.step();
    stmt.finalize();
    const rows = db.exec('SELECT hex(id) AS id FROM objstore LIMIT 1');
    db.exec('DELETE FROM objstore');
    db.exec('COMMIT');
    const id = rows && rows[0] ? rows[0].id : null;
    postMessage({ type: 'roundtrip', ok: true, id });
}

self.onmessage = async (event) => {
    try {
        const { type, payload } = event.data || {};
        switch (type) {
            case 'init':
                await ensureDatabase(payload);
                postMessage({ type: 'ready' });
                break;
            case 'roundtrip':
                await handleRoundTrip(payload);
                break;
            default:
                postMessage({
                    type: 'error',
                    message: `Unknown message type: ${type}`,
                });
        }
    } catch (err) {
        postMessage({
            type: 'error',
            message: err && err.message ? err.message : String(err),
        });
    }
};

