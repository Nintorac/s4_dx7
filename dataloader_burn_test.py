# %%
from itertools import count
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.data").setLevel(logging.ERROR)
logging.getLogger("ray.data._internal.execution.streaming_executor").setLevel(logging.ERROR)

# logger.setLevel(logging.ERROR)
from s4_dx7.lib.data.audio_data_module import AudioDataModule
import ray
ray.init(log_to_driver=False)
bit_rate=8
sr=8000
sample_size=8
data_module = AudioDataModule(bit_rate=bit_rate, sr=sr, limit=6400)
# Iterate over each batch (assuming a single batch here for simplicity)
for i in tqdm(count()):
    c=count()
    for batch in tqdm(data_module.get_train_dataloader(sample_size), leave=False):
        if next(c)>1000:
            break

# %%
 