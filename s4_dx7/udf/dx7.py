from typing import List
import duckdb
from duckdb.typing import BLOB, DuckDBPyType, BIGINT
from s4_dx7.lib.dx7 import clean_voice
from s4_dx7.lib.dx7 import extract_voices



VOICES_TYPE = DuckDBPyType(list[{'bytes': bytes, 'index':int}])

def extract_voices_udf(cart_bytes: BLOB)->VOICES_TYPE:
    cart_voices = []
    for row in cart_bytes.to_pandas():
        extracted = extract_voices(row)
        cart_voices.append([{'index':i, 'bytes': bytes(sysex_data)} for i, sysex_data in enumerate(extracted)])
    return cart_voices

def clean_voice_udf(voice_bytes: BLOB) -> BLOB:
    voices_df = voice_bytes.to_pandas()
    return voices_df.map(clean_voice)


