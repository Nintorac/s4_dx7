#%%

from itertools import count
import duckdb
import ray
from tqdm import tqdm
from s4_dx7.lib.render import render_batch
from s4_dx7.udf.render import render_midi_udf
import pandas as pd
from torch.utils.data import Dataset, DataLoader
# con = duckdb.connect()
# con.execute("attach 'audio.db'")
# con.create_function('render_midi', render_midi_udf, type='arrow')

# notes = con.execute('select * from audio."4_beat_phrases" limit 3').df()
# con.execute('select render_midi(notes) from notes').df()

class DX7Patch2PatchDataset(Dataset):

    table = 'audio."4_beat_phrases"'

    def __init__(self):

        con = self.get_conn()
        
        self._rows = con.execute(f'select rowid from {self.table}').df()
        # con.execute(f"create index rowid on {self.table} (rowid)")

    def get_conn(self):
        con = duckdb.connect()
        con.execute("attach 'audio.db' (READ_ONLY)")
        con.create_function('render_midi', render_midi_udf, type='arrow')

        return con
    
    def __getitem__(self, i):
        return self._rows.iloc[i]

    def __len__(self):
        return len(self._rows.index)
    
    def collate_fn(self, rowids):
        rowids = pd.DataFrame(rowids)
        con = self.get_conn()
        data = con.query(
        f'''
        select notes from {self.table} 
        where rowid in (select rowid from rowids)
        ''').arrow()

        data = con.execute(
        f'''
        select render_midi(notes) from data
        ''').df()

        return data
if __name__=='__main__' and False:
    dataset = DX7Patch2PatchDataset()
    ds = ray.data.from_torch(dataset)
    from time import sleep
    from random import random
    import torch
    # ray.init()
    # torch.multiprocessing.set_start_method('spawn'),
    loader = ds.iter_torch_batches(
        batch_size=256, 
        local_shuffle_buffer_size=256, 
        # num_workers=0,
        collate_fn=dataset.collate_fn,
        prefetch_batches=8,
        # timeout=30
    )
    for j in tqdm(count()):
        for i, _ in enumerate(tqdm(loader, total=2000)):
            if i>2000: break
            sleep(random()/2+1)

# print(dataset.collate_fn([dataset[i] for i in range(10)]))
# %%
def float_to_pcm16(audio):
    import numpy

    ints = (audio * 32767).astype(numpy.int16)
    # little_endian = ints.astype('<u2')
    # buf = little_endian.tostring()
    return ints


def read_pcm16(path):
    import soundfile

    audio, sample_rate = soundfile.read(path)
    assert sample_rate in (8000, 16000, 32000, 48000)
    pcm_data = float_to_pcm16(audio)
    return pcm_data, sample_rate

if __name__=='__main__':

    import ray
    import sqlite3
    import duckdb
    from s4_dx7.lib.render import render_batch
    import pyarrow as pa
    from tqdm import tqdm
    from pedalboard_native.io import AudioFile
    import io
    import numpy as np
    def create_connection():
        return duckdb.connect("audio.db")


    # ds = ray.data.
    ds = ray.data.read_sql(
        f'SELECT * FROM "4_beat_phrases" limit 128',
        create_connection,
        parallelism=1
    )



    def thing(batch):
        batch_table = pa.Table.from_pydict(batch)
        signals = ray.get(render_batch.remote(batch_table['notes']))
        
        return {'signals': signals}
        # return batch
    ds = ds.map_batches(thing, zero_copy_batch=True, batch_size=128, num_cpus=0.1)

    batches = [x for x in tqdm(ds.iter_torch_batches(batch_size=8, prefetch_batches=2))]

#%%
from s4_dx7.lib.models.mamba_simple import ModelArgs as MambaArgs, Mamba
args = MambaArgs(
    1,
    4,
    2**16
)
mamba = Mamba(args)
mamba(batches[0]['signals'][:, :1000])
