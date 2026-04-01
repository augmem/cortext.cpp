-- fixture: idempotent-put
DROP TABLE IF EXISTS objstore;
CREATE VIRTUAL TABLE objstore USING objstore();
WITH first AS (
    SELECT objstore_put(zeroblob(16)) AS id
),
second AS (
    SELECT objstore_put(zeroblob(16)) AS id
)
SELECT wasm_expect('idempotent-put/same-id', (SELECT id FROM first) = (SELECT id FROM second), 1);
WITH first AS (
    SELECT objstore_put(zeroblob(16)) AS id
),
payload AS (
    SELECT objstore_put(randomblob(16)) AS id
)
SELECT wasm_expect('idempotent-put/different', (SELECT id FROM first) = (SELECT id FROM payload), 0);
