-- fixture: crud-basic
DROP TABLE IF EXISTS objstore;
CREATE VIRTUAL TABLE objstore USING objstore();
INSERT INTO objstore(data) VALUES (x'0102030405060708');
SELECT wasm_expect('crud-basic/row-count', EXISTS(SELECT 1 FROM objstore), 1);
INSERT INTO objstore(data) VALUES (zeroblob(512));
SELECT wasm_expect(
    'crud-basic/payload-bytes',
    (SELECT length(data) FROM objstore ORDER BY rowid DESC LIMIT 1),
    512
);
SELECT wasm_expect('crud-basic/exists', objstore_exists(id), 1)
FROM objstore
LIMIT 1;
DELETE FROM objstore;
SELECT wasm_expect('crud-basic/empty', EXISTS(SELECT 1 FROM objstore), 0);
