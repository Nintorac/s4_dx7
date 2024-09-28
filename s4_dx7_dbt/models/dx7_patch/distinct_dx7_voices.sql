select DISTINCT voice_id, bytes 
from {{ ref('clean_dx7_voices') }}