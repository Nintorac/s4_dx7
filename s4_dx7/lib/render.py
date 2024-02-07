from functools import partial
import io
from itertools import chain
import json
import os
import random
from tempfile import NamedTemporaryFile
from typing import List, Optional

from mido import Message
import numpy
from pedalboard import load_plugin
from torchaudio.transforms import Resample
from librosa import resample
from tqdm import tqdm
import pydub
import ray
import pyarrow as pa
import duckdb
import mido

from s4_dx7.lib.dx7 import DX7Single, consume_syx, dx7_bulk_pack

RENDER_SAMPLE_RATE = 44100

def float_to_pcm16(audio, bit_rate=16):

    ints = ((audio + 1) * (2**(bit_rate-1))).astype(numpy.int32)

    return ints


def to_midi(notes: List[dict], transpose:int=0, bpm:int=120)->List[Message]:    
	# the note's are represented as start_time, duration. 
    # But MIDI needs note_on, note_off tuples, 
    # so each note event will produce two MIDI messages
    notes = json.loads(notes)
    time_multiplier = 60/bpm
    return list(chain(*[
            (Message('note_on', time=time_multiplier*(note['start_time']), note=note['note']+transpose, velocity=note['velocity']), 
             Message('note_off', time=time_multiplier*(note['start_time']+note['duration']), note=note['note']+transpose, velocity=0)
             ) for note in notes]))

# @ray.remote(num_cpus=1, memory=500 * 1024 * 1024) #5mb
def render_batch(notes: List[List[dict]], sr: int, duration: float, voice:Optional[bytes]=None, plugin: bytes=None, bpm: int=120)->pa.array:
    chunk = notes.to_pandas()
    # Load a VST3 or Audio Unit plugin from a known path on disk:
    if plugin:
        with NamedTemporaryFile('r+b') as f:
            f.write(plugin)
            f.flush()
            instrument = load_plugin(f.name)
    else:
        instrument = load_plugin(os.environ['DEFAULT_INSTRUMENT_PATH'])
    syx = []
    if voice:
        with NamedTemporaryFile('r+b') as f:
            f.write(voice)
            f.flush()
            syx = mido.read_syx_file(f.name)

    samples = []
    # t = tqdm()

    # set the voice patch
    instrument(
        syx,
        duration=0.1, # seconds
        sample_rate=sr,
    )

    to_midi_f = partial(to_midi, bpm=bpm)
    for notes in map(to_midi_f, chunk):
        import time
        from datetime import datetime
        import logging
        logger=logging.getLogger('lol')
        logger.setLevel(logging.INFO)
        st = time.time()
        print(f'starting... {datetime.now()})')

        x = instrument(
            notes,
            duration=duration, # seconds
            sample_rate=RENDER_SAMPLE_RATE,
        )
        print(time.time()-st)
        x = resample(x, orig_sr=RENDER_SAMPLE_RATE, target_sr=sr)
        samples.append(float_to_pcm16(x.mean(0)))
        # t.update(1)

    del instrument
    import gc
    gc.collect()

    # 1/0
    return pa.array(samples)

