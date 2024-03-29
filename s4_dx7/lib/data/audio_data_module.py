import os
import duckdb
import pyarrow as pa
import ray
import numpy as np
import torch

from s4_dx7.lib.render import render_batch

def create_connection():
    return duckdb.connect(os.environ['AUDIO_DB'])

class AudioDataModule():
    def __init__(self, table='melodies', bit_rate=16, sr=44100, duration=2.5, limit=None):
        print(bit_rate, sr, duration)

        def dataset(dataset_type):

            delta_rate = 16-bit_rate-1
            source_patch, target_patch = self.get_voices()
            limit_str = '' if limit is None else f'limit {limit}'
            ds = ray.data.read_sql(
                f'SELECT rowid FROM "{table}" order by rowid {limit_str}',
                create_connection,
                parallelism=1
            )
            train, test = ds.train_test_split(test_size=0.2)
            test, validate = test.train_test_split(test_size=0.5)

            def f(batch):
                conn = create_connection()
                batch_table = conn.execute(f'''
                select notes from "{table}"
                where rowid in (select rowid from batch)
                ''').arrow()
                # for _ in range(3):
                    # batch_table = pa.Table.from_pydict(batch)
                signals_source = render_batch(batch_table['notes'], sr, duration + 1/sr, voice=source_patch)
                signals_target = render_batch(batch_table['notes'], sr, duration + 1/sr, voice=target_patch)
                    # try:
                    #     signals_target, signals_source = ray.get([signals_target, signals_source], timeout=128)
                    #     break
                    # except Exception as e:
                    #     ray.kill([signals_target, signals_source])
                    #     continue



                signals_target = self.clean_signal(signals_target, delta_rate, bit_rate)
                signals_source = self.clean_signal(signals_source, delta_rate, bit_rate)

                # print(max(signals_source.max(), signals_target.max()))
                # print(min(signals_source.min(), signals_target.min()))
                
                # return {"x": signals_source, "y": signals_target.unsqueeze(-1)}
                return {"x": signals_source[:,:-1], "y": signals_target[:,1:].unsqueeze(-1)}
                # return {"x": signals_source[:,:-1], "y": signals_source[:,1:].unsqueeze(-1)}

            pipeline = lambda dataset: dataset.repartition(200).map_batches(f, zero_copy_batch=True, batch_size=128)
            if dataset_type=='train':
                return pipeline(train)
            elif dataset_type=='test':
                return pipeline(test)
            else: 
                return pipeline(validate)

        self._dataset = dataset
        self._tl = None
        self._el = None
        self._tel = None

    @staticmethod
    def clean_signal(signals, delta_rate, bit_rate):                

        signals = torch.tensor(signals.to_pylist())
        bit_crushed = signals//(2**delta_rate)
        bit_crushed = bit_crushed.clamp(0, 2**bit_rate-1)
        return bit_crushed

    @staticmethod
    def get_voices():
        target_patch, = create_connection().execute("select bytes from dx7_voices where path like '%SynprezFM_02%' and index=0").df().bytes
        source_patch, = create_connection().execute("select bytes from dx7_voices where path like '%siine%' and index=0").df().bytes
        
        return source_patch, target_patch

    def get_train_dataloader(self, batch_size):
        if self._tl is None:
            self._tl = self._dataset('train').iter_torch_batches(
                batch_size=batch_size,
                # prefetch_batches=2
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
