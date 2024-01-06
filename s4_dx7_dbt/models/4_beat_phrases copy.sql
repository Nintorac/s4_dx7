select * from 
FROM {{ source('midi', 'detailed_notes') }} e
where midi_id like 'f%'