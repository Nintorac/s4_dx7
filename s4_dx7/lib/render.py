import io
from itertools import chain
import json
from tempfile import NamedTemporaryFile
from typing import List

from mido import Message
from pedalboard import load_plugin
from tqdm import tqdm
import pydub
import ray
import pyarrow as pa
import duckdb


SAMPLE_RATE = 22050

def float_to_pcm16(audio):
    import numpy

    ints = ((audio + 1) * 32767).astype(numpy.int32)

    return ints


def to_midi(notes: List[dict], transpose:int=0)->List[Message]:    
	# the note's are represented as start_time, duration. 
    # But MIDI needs note_on, note_off tuples, 
    # so each note event will produce two MIDI messages
    notes = json.loads(notes)
    return list(chain(*[
            (Message('note_on', time=note['start_time'], note=note['note']+transpose, velocity=note['velocity']), 
             Message('note_off', time=note['start_time']+note['duration'], note=note['note']+transpose, velocity=0)
             ) for note in notes]))


@ray.remote(num_cpus=1)
def render_batch(notes: List[List[dict]])->pa.array:
    chunk = notes.to_pandas()
    # Load a VST3 or Audio Unit plugin from a known path on disk:
    import os
    instrument = load_plugin("instruments/Dexed.vst3")
    samples = []
    # t = tqdm()
    print(len(chunk))
    for notes in map(to_midi, chunk):
        # raise ValueError(notes)
        # print(1)
        x = instrument(
        notes,
        duration=2.5, # seconds
        sample_rate=SAMPLE_RATE,
        )
        # print(x.shape)
        samples.append(float_to_pcm16(x.mean(0)))
        # t.update(1)

    # 1/0
    return pa.array(samples)

