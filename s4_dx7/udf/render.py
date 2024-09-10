import io
from itertools import chain
from tempfile import NamedTemporaryFile
from typing import List

import pyarrow
from mido import Message
from pedalboard import load_plugin
from tqdm import tqdm
import pydub
import ray
import pyarrow as pa
import duckdb
from duckdb.typing import VARCHAR, DuckDBPyType, BLOB, INTEGER, FLOAT

from s4_dx7.lib.render import render_batch
from s4_dx7.udf.digital import AUDIO_TYPE

# AUDIO = DuckDBPyType(List[str])
def render_midi_udf(batch: VARCHAR)->BLOB:
    
    rows_per_batch=128
    # rows_per_batch=max(10, batch.length()//32) # max of size 64 batches @ 2048 sized vectors
    chunks = []
    for chunk_start in range(0, batch.length(), rows_per_batch):
        
        chunk = batch.slice(chunk_start, rows_per_batch)
        chunk = render_batch.remote(chunk)
        chunks.append(chunk)

    return pa.concat_arrays(ray.get(chunks))

def render_dx7_udf(
        notes: VARCHAR,
        voice: BLOB,
        sr: INTEGER,
        duration: FLOAT
    )->AUDIO_TYPE:
    """
    probably not super performant at scale but good enough for training
    """
    promises = []
    f = ray.remote(num_cpus=1, memory=500 * 1024 * 1024)(render_batch)
    a = lambda x: pyarrow.array([x])
    for n, v, s, d in zip(notes, voice, sr, duration):
        promise = f.remote(a(n), s.as_py(), d.as_py(), v.as_py(), pcm_encode=False)
        promises.append(promise)

    # return pa.array(ray.get(promises))
    results = ray.get(promises)
    return pa.concat_arrays(results)
