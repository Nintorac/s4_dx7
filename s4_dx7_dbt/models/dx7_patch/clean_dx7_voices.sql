with clean_voices as (
    select 
        path
        , index
        , clean_voice(bytes) bytes
    from {{ ref('dx7_voices') }}
)
select 
    md5(to_base64(bytes)) voice_id
    , *
from clean_voices