with voices as (
SELECT c.voice_id, c.bytes, path, index from {{ ref('distinct_dx7_voices') }} d
join {{ ref('clean_dx7_voices') }} c
on c.voice_id=d.voice_id
)
select
    voice_id,
    bytes,
    list({path: path, index: index}) metadata
from voices
GROUP BY voice_id, bytes