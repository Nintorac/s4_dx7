select 
    distinct to_json(list({
            'start_time': e.start_time-floor(e.start_time/4)*4,
            'note': e.note,
            'duration': e.duration,
            'velocity': e.velocity
        })) notes

from {{ ref("4_beat_phrases") }} e
where phrase_id in (
    --- phrase filter for melodies here
    -- at least 3 notes
    -- at least 3 beats of the phrase filled
    -- maximum 1 polpyhony
    select phrase_id from {{ ref("phrase_stats") }}
    where polyphony=1 
    and sum_duration>3 
    and n_notes>=4
    and n_notes<=18 
    and min_note > 21  -- A0
    and max_note < 108 -- C8
    and unique_notes > 6
) 
group by phrase_id