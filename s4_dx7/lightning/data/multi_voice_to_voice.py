from enum import StrEnum
from functools import partial
import os
from typing import Optional
import duckdb
import pyarrow as pa
from pydantic import BaseModel
import ray
import numpy as np
import torch
from torchaudio.functional import mu_law_encoding
from s4_dx7.lib.render import render_batch
from s4_dx7.udf import Plugin
from s4_dx7.utils import get_duckdb_connection
import ray

ctx = ray.data.DataContext.get_current()
ctx.use_push_based_shuffle = True

def create_connection():
    duckdb.execute('install sqlite')
    duck = duckdb.connect(os.environ['AUDIO_DB'])
    Plugin.configure_connection(None, duck)

    return duck

class DataSplit(StrEnum):
    TRAIN = 'train'
    TEST = 'test'
    VALIDATE = 'validate'

class PipelineConfiguration(BaseModel):
    partitions: Optional[int] = 10
    f_batch_size: Optional[int] = 16
    f_concurrency: Optional[int] = 4
    override_read_blocks: Optional[int] = None
    f_num_cpus: int = 1
    loader_prefetch: Optional[int] = None

class MultiVoice2VoiceDataModule():
    def __init__(self,
        table='multivoice_dataset',
        bit_rate=16,
        sr=44100,
        duration=2.5,
        limit=None,
        patch_baud_rate=300,
        pipeline_config: PipelineConfiguration = None
    ):
        print(bit_rate, sr, duration)

        if pipeline_config is None: pipeline_config = PipelineConfiguration()
        self.pipeline_config = pipeline_config
        self.bit_rate = bit_rate
        self.duration = duration
        self.sr = sr
        self.table = table
        self.limit = limit
        self.patch_baud_rate = patch_baud_rate
        self._tl = None
        self._el = None
        self._tel = None

    def _dataset(self, data_split):

            limit_str = '' if self.limit is None else f'limit {self.limit}'
            data_split = DataSplit(data_split)
            ds = ray.data.read_sql(
                f'''
                SELECT rowid
                FROM "{self.table}"
                where data_split='{data_split.value}'
                order by rowid {limit_str}
                ''',
                create_connection,
                override_num_blocks=self.pipeline_config.override_read_blocks
            )


            f = partial(self.f,
                table=self.table,
                sr=self.sr,
                duration=self.duration,
                bit_rate=self.bit_rate,
                baud_rate=self.patch_baud_rate)
            if data_split == DataSplit.TRAIN:
                ds = ds.random_shuffle()
            if self.pipeline_config.partitions:
                ds = ds.repartition(self.pipeline_config.partitions, shuffle=True)

            pipeline = ds.map_batches(
                f, 
                zero_copy_batch=True, 
                batch_format='pandas',
                batch_size=self.pipeline_config.f_batch_size, 
                num_cpus=self.pipeline_config.f_num_cpus,
                concurrency=self.pipeline_config.f_concurrency
            )
            # raise ValueError(pipeline.stats())
            return pipeline
            # return ds


        # self._dataset = dataset
    @staticmethod
    def f(batch, table, sr, duration, bit_rate, baud_rate):

        conn = create_connection()
        # raise ValueError(batch)
        batch_table = conn.execute(f'''
            with batch_data as (
                select * from "{table}"
                where rowid in (select rowid from batch)
            )
            select
                source_voice_id
                , target_voice_id
                , phrase_id
                , render_dx7(
                    midi_notes
                    , source_voice_bytes
                    , {sr}
                    , {duration}
                ) source_voice ,
                render_dx7(
                    midi_notes
                    , target_voice_bytes
                    , {sr}
                    , {duration}
                ) target_voice ,
                fsk_encode_syx(target_voice_bytes, {sr}, {baud_rate}) target_encoding
            from batch_data
        ''').arrow()
        
        target_encoding = torch.tensor(batch_table['target_encoding'].to_pylist())
        source_voice = torch.tensor(batch_table['source_voice'].to_pylist())
        target_voice = torch.tensor(batch_table['target_voice'].to_pylist())

        signals_source = torch.cat([target_encoding, source_voice], dim=-1)
        signals_target = torch.cat([torch.zeros_like(target_encoding), target_voice], dim=-1)

        signals_target = mu_law_encoding(signals_target, bit_rate)
        signals_source = mu_law_encoding(signals_source, bit_rate)

        # raise ValueError(signals_source.shape)
        # raise ValueError(signals_source.shape, signals_target.shape)
        return {
            "x": signals_source,
            "y": signals_target,
            "encoding": target_encoding,
            # "source_voice_id": batch_table['source_voice_id'].to_pylist(),
            "rowid": batch.rowid
            # "target_voice_id": batch_table['target_voice_id'].to_pylist(),
            # "phrase_id": batch_table['phrase_id'].to_pylist(),
        }


    def get_train_dataloader(self, batch_size):
        if self._tl is None:
            self._tl = self._dataset('train').iter_torch_batches(
                batch_size=batch_size,
                prefetch_batches=self.pipeline_config.loader_prefetch
            )
        return self._tl

    def get_eval_dataloader(self, batch_size):
        

        if self._el is None:
            self._el = self._dataset('validate').iter_torch_batches(
                batch_size=batch_size,
                # prefetch_batches=2
            )
        return self._el
    
    def get_test_dataloader(self, batch_size):

        if self._tel is None:
            self._tel = self._dataset('test').iter_torch_batches(
                batch_size=batch_size,
                # prefetch_batches=2
            )
        return self._tel
