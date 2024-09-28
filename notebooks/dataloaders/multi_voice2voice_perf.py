# %%
from datetime import datetime
from time import sleep
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import numpy as np
import pandas as pd
import scipy.signal as s
from s4_dx7.lightning.data import MultiVoice2VoiceDataModule
from s4_dx7.lib.visualistaion.audio import waveform_segment_figure
from torchaudio.functional import mu_law_decoding
from s4_dx7.lightning.data.multi_voice_to_voice import PipelineConfiguration
from s4_dx7.notebook.mel_spec_audio import plot_melspectrogram_and_play_button
from tqdm import tqdm_notebook
## TODO figure out how to remove error ouput from rendered document
import ray
#%%
ray.init(_temp_dir='/app/ray')
#%%
for i in tqdm_notebook(range(3)): pass
#%%
# Define a helper function to plot the mel spectrogram and create the play button
bit_rate=8
sr=40000
baud=8000
sample_size=10000
batch_size=2

dt = int(1/baud*sr)  # samples per bit
encoding_duration = (dt*8*155)/sr
encoding_samples = dt*8*155
pipe_config = PipelineConfiguration(
    f_batch_size=None,
    partitions=5000,
    f_concurrency=3,
    override_read_blocks=1,
    f_num_cpus=1,
    loader_prefetch = None
)


data_module = MultiVoice2VoiceDataModule(
    bit_rate=bit_rate,
    limit=sample_size,
    sr=sr,
    patch_baud_rate=baud,
    pipeline_config=pipe_config
)
# %%
pipeline = data_module._dataset('train')
# %%
# %%
times = [datetime.now().timestamp()]
for i, x in tqdm_notebook(enumerate(pipeline.iter_torch_batches(batch_size=batch_size))):
    sleep(0.5) # model.forward().backwared()
    times.append(datetime.now().timestamp())

    print(len(x['rowid']))
    if not i % 1000:
        plt.scatter((range(len(times))),np.gradient(np.array(times)))
        plt.show()
        plt.close('all')
    pass
# %%
print(pipeline.stats())
# %%
# %%
