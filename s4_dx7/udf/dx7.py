import duckdb
from duckdb.typing import BLOB, DuckDBPyType
import mido
from s4_dx7.lib.dx7 import DX7Single, consume_syx
from tempfile import NamedTemporaryFile


VOICES_TYPE = DuckDBPyType(list[{'bytes': bytes, 'index':int}])

def sysex_bytes(sysex):
    with NamedTemporaryFile('rb') as f:
        mido.write_syx_file(f.name, [sysex])
        sysex_bytes = f.read()
        return sysex_bytes

def extract_voices(cart_bytes: bytes): 
    with NamedTemporaryFile('r+b') as f:
        f.write(cart_bytes)
        return [sysex_bytes(DX7Single.to_syx([x.values()])) for x in consume_syx(f.name) if x is not None]


def extract_voices_udf(cart_bytes: BLOB)->VOICES_TYPE:
    cart_voices = []
    for row in cart_bytes.to_pandas():
        extracted = extract_voices(row)
        cart_voices.append([{'index':i, 'bytes': bytes(sysex_data)} for i, sysex_data in enumerate(extracted)])
    return cart_voices