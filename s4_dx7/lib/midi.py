import mido
from tempfile import NamedTemporaryFile


def sysex_bytes(sysex: mido.Message):
    with NamedTemporaryFile('rb') as f:
        mido.write_syx_file(f.name, [sysex])
        sysex_bytes = f.read()
        return sysex_bytes