select voice_id, fsk_encode_syx(bytes) bytes
from {{ ref('distinct_dx7_voices') }}