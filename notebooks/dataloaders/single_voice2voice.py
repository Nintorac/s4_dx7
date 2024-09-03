# %%
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import numpy as np
import pandas as pd
from s4_dx7.lightning.data.single_voice_to_voice import SingleVoice2VoiceDataModule
from s4_dx7.lib.visualistaion.audio import waveform_segment_figure
from s4_dx7.notebook.mel_spec_audio import plot_melspectrogram_and_play_button
#%%
# Define a helper function to plot the mel spectrogram and create the play button
bit_rate=16
sr=8000
sample_size=15
data_module = SingleVoice2VoiceDataModule(bit_rate=bit_rate, sr=sr, limit=sample_size)
# Iterate over each batch (assuming a single batch here for simplicity)
for batch in data_module.get_train_dataloader(sample_size):
    batch_size = batch['x'].shape[0]

    for i in range(2):
        # Select the i-th sample from both 'x' and 'y' signals in the batch
        x_signal = batch['x'][i]
        y_signal = batch['y'][i].squeeze(-1)
        # raise ValueError(y_signal.min(), y_signal.max())
        y_signal = (y_signal.float() / (2**(bit_rate-1)))-1
        x_signal = (x_signal.float() / (2**(bit_rate-1)))-1
        
        # Plot and display x
        print(f"Playing and displaying x[{i}]")
        plot_melspectrogram_and_play_button(x_signal, sr)

        # plot and display y
        print(f"playing and displaying y[{i}]")
        plot_melspectrogram_and_play_button(y_signal, sr)
    break

# %%

waveform_segment_figure(x_signal, 8000, 0, 0.2)
waveform_segment_figure(y_signal, 8000, 0, 0.2)
# %%
