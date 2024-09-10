#%% [markdown]
# This page is a render of the notebook found at `notebooks/dataloaders/multi_voice2voice.py` in the [source repo](https://github.com/Nintorac/s4_dx7)
# 
# > [!warning]
# > **The audio can be quite loud**: Turn down your volume
# 
# It shows several random examples of data used in the training of [[s4-dx7-vc-fir-03]]
# %%
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
from s4_dx7.notebook.mel_spec_audio import plot_melspectrogram_and_play_button
## TODO figure out how to remove error ouput from rendered document
#%%
#%%
# Define a helper function to plot the mel spectrogram and create the play button
bit_rate=8
sr=40000
baud=8000
sample_size=15

dt = int(1/baud*sr)  # samples per bit
encoding_duration = (dt*8*155)/sr
encoding_samples = dt*8*155

data_module = MultiVoice2VoiceDataModule(bit_rate=bit_rate, limit=sample_size, sr=sr, patch_baud_rate=baud)
# Iterate over each batch (assuming a single batch here for simplicity)
for batch in data_module.get_train_dataloader(sample_size):
    batch_size = batch['x'].shape[0]

    for i in range(5):
        # Select the i-th sample from both 'x' and 'y' signals in the batch
        x_signal = batch['x'][i].cpu()
        y_signal = batch['y'][i].squeeze(-1).cpu()
        # raise ValueError(y_signal.min(), y_signal.max())
        y_signal = mu_law_decoding(y_signal, bit_rate)
        x_signal = mu_law_decoding(x_signal, bit_rate)
        
        # print(f"source voice - {batch['source_voice_id'][i]}")
        # print(f"target voice - {batch['target_voice_id'][i]}")
        # print(f"phrase - {batch['phrase_id'][i]}")
        # Plot and display encoding
        print(f"Playing and displaying encoding")
        plot_melspectrogram_and_play_button(batch['encoding'][i].cpu(), sr)

        # Plot and display x
        print(f"Playing and displaying x[{i}]")
        plot_melspectrogram_and_play_button(x_signal, sr)

        # plot and display y
        print(f"playing and displaying y[{i}]")
        plot_melspectrogram_and_play_button(y_signal, sr)

        wave_fig = waveform_segment_figure(
            x_signal,
            sr,
            (
                (0, (dt*16*8)/sr), # first 16 bytes of encoding
                (encoding_duration, encoding_duration+1), # 1s of audio
                (encoding_duration+1, encoding_duration+1.1), # 0.1s of audio
                (encoding_duration+1, encoding_duration+1.01), # 0.01s of audio
            ),
            title="source waveforms - first 16 bytes of encoding, 1s of audio, 0.1s of audio, 0.01s of audio"
        )
        display(wave_fig)
        plt.close("all") # supress inlined plots https://stackoverflow.com/questions/49545003/how-to-suppress-matplotlib-inline-for-a-single-cell-in-jupyter-notebooks-lab
        wave_fig = waveform_segment_figure(
            y_signal,
            sr,
            (
                (encoding_duration, encoding_duration+1), # first second of audio source
                (encoding_duration+1, encoding_duration+1.1), # 0.1s of audio
                (encoding_duration+1, encoding_duration+1.01), # 0.01s of audio
            ),
            title="target waveforms - 1s of audio, 0.1s of audio, 0.01s of audio"
        )
        display(wave_fig)
        plt.close("all") # supress inlined plots https://stackoverflow.com/questions/49545003/how-to-suppress-matplotlib-inline-for-a-single-cell-in-jupyter-notebooks-lab

        t = np.arange(0, x_signal.shape[0] / sr, 1.0 / sr)
        f, t_s, Zxx = s.stft(x_signal, nfft=128, fs=sr, nperseg=4, noverlap=3, padded=True)

        plt.pcolormesh(t_s[:encoding_samples//8], f, np.abs(Zxx[...,:encoding_samples//8]))
        plt.title(f"{encoding_samples//8} (~19 bytes) samples of the patch encoding as a spectrogram")
        plt.show()

        plt.close("all") # supress inlined plots https://stackoverflow.com/questions/49545003/how-to-suppress-matplotlib-inline-for-a-single-cell-in-jupyter-notebooks-lab
    break

# %%
