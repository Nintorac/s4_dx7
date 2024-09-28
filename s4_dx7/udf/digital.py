from typing import List
import duckdb
from duckdb.typing import BLOB, DuckDBPyType, BIGINT
from s4_dx7.lib.audio.digital import fsk_encode_syx
from s4_dx7.lib.dx7 import clean_voice
from s4_dx7.lib.dx7 import extract_voices

AUDIO_TYPE = DuckDBPyType(list[float])
def fsk_encode_syx_udf(bytes_to_encode: BLOB, sampling_rate: BIGINT, baud_rate: BIGINT)->AUDIO_TYPE:
    if not len(set(sampling_rate))==len(set(baud_rate))==1:
        raise ValueError('cant process different sample/baud rates in a single batch')
    result = fsk_encode_syx(bytes_to_encode.to_pandas(), sampling_rate[0].as_py(), baud_rate[0].as_py())
    result = list(result.numpy())
    return result
