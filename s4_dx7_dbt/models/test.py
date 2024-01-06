import io
from itertools import chain
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

def to_midi(notes: List[dict], transpose:int=0)->List[Message]:    
	# the note's are represented as start_time, duration. 
    # But MIDI needs note_on, note_off tuples, 
    # so each note event will produce two MIDI messages
    return list(chain(*[
            (Message('note_on', time=note['start_time'], note=note['note']+transpose, velocity=note['velocity']), 
             Message('note_off', time=note['start_time']+note['duration'], note=note['note']+transpose, velocity=0)
             ) for note in notes]))


@ray.remote(num_cpus=1)
def process_chunk(notes: List[List[dict]])->pa.array:

    chunk = notes.to_pandas()
    # Load a VST3 or Audio Unit plugin from a known path on disk:
    import os
    instrument = load_plugin("../instruments/Dexed.vst3")
    samples = []
    # t = tqdm()
    for notes in map(to_midi, chunk):
        
        x = instrument(
        notes,
        duration=5, # seconds
        sample_rate=SAMPLE_RATE,
        )

        x = pydub.AudioSegment(
            x.tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=x.dtype.itemsize,
            channels=2
        )
        # with NamedTemporaryFile('rb+') as f:
        with io.BytesIO() as f:
            x = x.export(f, format='flac')
            x = f.read()
        samples.append(x)
        # t.update(1)

    return pa.array(samples)

def process_batch(batch: pa.array)->pa.array:
    
    rows_per_batch=max(10, batch.length()//32) # max of size 64 batches @ 2048 sized vectors
    chunks = []
    for chunk_start in range(0, batch.length(), rows_per_batch):
        
        chunk = batch.slice(chunk_start, rows_per_batch)
        chunk = process_chunk.remote(chunk)
        chunks.append(chunk)

    return pa.concat_arrays(ray.get(chunks))


def model(dbt, session):
    dbt.config('')
    import pandas as pd
    
    return pd.DataFrame([{'a': 1}, {'a': 2}])