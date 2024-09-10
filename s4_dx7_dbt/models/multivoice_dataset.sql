with dataset as (
    select
        voice_id target_voice_id
        , 'e7596651f46069a69e2773b7b004f9ae' source_voice_id
        , phrase_id
        , (
            select bytes
            from {{ ref('distinct_dx7_voices') }}
            -- sine patch
            where voice_id='e7596651f46069a69e2773b7b004f9ae'
        ) source_voice_bytes
        , bytes target_voice_bytes
        , notes midi_notes
        , CASE
            WHEN NTILE(10) OVER (ORDER BY voice_id) < 6 THEN 'train'
            WHEN NTILE(10) OVER (ORDER BY voice_id) < 8 THEN 'test'
            ELSE 'validate'
        END AS data_split
    FROM {{ ref('melodies') }} m
    cross join {{ ref('distinct_dx7_voices') }} v
)
select
    md5(target_voice_id || source_voice_id || phrase_id) dataset_item_id
    , *
from dataset
order by dataset_item_id