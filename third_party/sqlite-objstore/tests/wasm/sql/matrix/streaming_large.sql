-- fixture: streaming-large
DROP TABLE IF EXISTS objstore;
CREATE VIRTUAL TABLE objstore USING objstore();
WITH payload AS (
    SELECT zeroblob(8 * 1024 * 1024) AS bytes
)
INSERT INTO objstore(data)
SELECT bytes FROM payload;
SELECT wasm_expect(
    'streaming-large/roundtrip',
    (SELECT length(data) FROM objstore LIMIT 1),
    8 * 1024 * 1024
);
