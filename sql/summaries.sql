-- One row per subject x channel: window features averaged within the subject.
-- Group tests use this, so one subject = one data point.
CREATE OR REPLACE VIEW subject_channel AS
SELECT
    f.subject_id,
    s."group",
    f.channel,
    count(*)          AS n_windows,
    avg(f.delta)      AS delta,
    avg(f.theta)      AS theta,
    avg(f.alpha)      AS alpha,
    avg(f.beta)       AS beta,
    avg(f.rel_delta)  AS rel_delta,
    avg(f.rel_theta)  AS rel_theta,
    avg(f.rel_alpha)  AS rel_alpha,
    avg(f.rel_beta)   AS rel_beta,
    avg(f.tbr)        AS tbr
FROM features f
JOIN subjects s USING (subject_id)
GROUP BY f.subject_id, s."group", f.channel;

-- One row per group x channel: window counts and TBR summaries (windows pooled).
CREATE OR REPLACE VIEW group_channel_summary AS
SELECT
    s."group",
    f.channel,
    count(*)                       AS n_windows,
    count(DISTINCT f.subject_id)   AS n_subjects,
    avg(f.tbr)                     AS mean_tbr,
    median(f.tbr)                  AS median_tbr
FROM features f
JOIN subjects s USING (subject_id)
GROUP BY s."group", f.channel;
