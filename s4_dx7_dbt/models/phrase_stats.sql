select
    coalesce(s.phrase_id, p.phrase_id) phrase_id
    , s.* exclude (phrase_id)
    , polyphony 
from {{ ref("simple_phrase_stats") }} s
left outer JOIN {{ ref("phrase_polyphonies") }} p
on s.phrase_id=p.phrase_id