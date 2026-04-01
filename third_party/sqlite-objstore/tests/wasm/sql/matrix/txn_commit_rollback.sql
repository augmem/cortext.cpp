-- fixture: txn-commit-rollback
DROP TABLE IF EXISTS objstore;
CREATE VIRTUAL TABLE objstore USING objstore();
BEGIN;
INSERT INTO objstore(data) VALUES (randomblob(64));
COMMIT;
SELECT wasm_expect('txn-commit-rollback/after-commit', EXISTS(SELECT 1 FROM objstore), 1);
BEGIN;
INSERT INTO objstore(data) VALUES (randomblob(64));
ROLLBACK;
SELECT wasm_expect('txn-commit-rollback/after-rollback', EXISTS(SELECT 1 FROM objstore), 1);
DELETE FROM objstore;
SELECT wasm_expect('txn-commit-rollback/cleanup', EXISTS(SELECT 1 FROM objstore), 0);
