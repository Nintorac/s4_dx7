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
from duckdb.typing import VARCHAR, DuckDBPyType, BLOB

from s4_dx7.lib.render import render_batch

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
