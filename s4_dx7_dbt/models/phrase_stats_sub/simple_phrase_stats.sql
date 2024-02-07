select 
    phrase_id
    , sum(duration) sum_duration
    , count(*) n_notes
    , min(note) min_note
    , max(note) max_note
    , max_note-min_note note_range
    , count(DISTINCT note) unique_notes
from {{ ref("4_beat_phrases") }}
group by phrase_id