-- Subject-level 5-fold split, stratified by group.
-- Within each group, order subjects by a seeded hash and deal them into folds 0-4.
-- Every subject gets exactly one fold, so no subject is ever in both train and test.
CREATE OR REPLACE TABLE splits AS
SELECT
    subject_id,
    (row_number() OVER (PARTITION BY "group" ORDER BY hash(subject_id || '42')) - 1) % 5 AS fold
FROM subjects;
