-- This common table expression (CTE) computes the end time for each event
WITH events_with_endtimes AS (
    SELECT ROW_NUMBER() over (PARTITION by phrase_id order by start_time, note, velocity) id, phrase_id, start_time, start_time + duration AS end_time
    FROM {{ ref("4_beat_phrases") }}
)
-- This CTE generates all possible pairings of events with start and end times, considering only pairs within the same phrase
, events_cross_join AS (
    SELECT a.id AS a_id, b.id AS b_id, a.phrase_id AS phrase_id, 
           a.start_time AS a_start, a.end_time AS a_end, b.start_time AS b_start, b.end_time AS b_end
    FROM events_with_endtimes a
    CROSS JOIN events_with_endtimes b
    WHERE a.phrase_id = b.phrase_id AND a.id != b.id
)
-- This query selects only the overlapping events in the same phrase
, overlapping_events AS (
    SELECT phrase_id, a_id, b_id
    FROM events_cross_join
    WHERE NOT (a_end <= b_start OR a_start >= b_end)
)
-- This query counts the maximum number of overlaps for each event and phrase
, overlap_count AS (
    SELECT phrase_id, a_id id, COUNT(*) AS overlaps
    FROM overlapping_events 
    GROUP BY phrase_id, id
)
select phrase_id, max("overlaps") polyphony
from overlap_count
group by phrase_id