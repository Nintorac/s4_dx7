from collections import defaultdict
import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import numpy as np
import torch

from s4_dx7.lib.render import render_batch

# Function to create display the play button for audio
def create_play_button(audio_signal, sample_rate=22050):
    # Ensure the audio signal is numpy array in case it's a tensor or similar
    audio_signal = audio_signal if isinstance(audio_signal, np.ndarray) else audio_signal.detach().cpu().numpy()
    
    # the audio signal play button
    return Audio(data=audio_signal, rate=sample_rate)
    
# Function to compute the mel spectrogram and plot it
def create_melspec_figure(audio_signal, sample_rate=22050, title='Mel-frequency spectrogram'):
    # Ensure the audio signal is numpy array in case it's a tensor or similar
    audio_signal = audio_signal if isinstance(audio_signal, np.ndarray) else audio_signal.detach().cpu().numpy()

    # Compute the mel spectrogram
    S = librosa.feature.melspectrogram(y=audio_signal, sr=sample_rate, hop_length=sample_rate//40, win_length=sample_rate//20)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot the mel spectrogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    spec = librosa.display.specshow(S_dB, sr=sample_rate, x_axis='time', y_axis='mel', ax=ax, win_length=sample_rate//20, hop_length=sample_rate//40)

    fig.colorbar(spec, format='%+2.0f dB', ax=ax)

    ax.set_title(title)
    fig.tight_layout()
    return fig


def render_piano_roll(messages):
    # Process messages to create start time, duration, and pitch for each note
    notes = []

    note_starts = defaultdict(lambda: None) # Dictionary to track note starts

    for msg in messages:
        time = msg.time  # accumulate time
        if msg.type == 'note_on' and msg.velocity > 0:
            note_starts[msg.note] = time
        elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)) and note_starts[msg.note] is not None:
            start_time = note_starts[msg.note]
            duration = time - start_time
            notes.append((start_time, duration, msg.note))
            note_starts[msg.note] = None  # Reset the note's start time

    # Plotting the piano roll
    fig, ax = plt.subplots()

    for start_time, duration, note in notes:
        ax.broken_barh([(start_time, duration)], (note, 0.8), facecolors='tab:blue')

    ax.set_xlabel('Time')
    ax.set_ylabel('Note')
    ax.set_title('Piano Roll')
    return fig

def render_voice(notes, voice, config):
    signals = render_batch(notes, config.sr, config.duration + 1/config.sr, voice=voice)

    # delta_rate = 16-config.bit_rate

    # def clean_signal(signals):                

    #     signals = torch.tensor(signals.to_pylist())
    #     bit_crushed = signals//(2**delta_rate)
    #     bit_crushed = bit_crushed.clamp(0, 2**config.bit_rate)
    #     return bit_crushed
    # signals = clean_signal(signals)

    return signals

def waveform_segment_figure(waveform, sample_rate, times, title='Waveform Segment'):
    """
    Plots a segment of a waveform using matplotlib.

    Parameters:
    - waveform: A 1D NumPy array containing the waveform data.
    - sample_rate: An integer representing the number of samples per second.
    - start_time: The start time in seconds to begin plotting.
    - stop_time: The stop time in seconds to end plotting.
    """
    # Plotting
    fig, axs = plt.subplots(len(times),1, figsize=(10, len(times)*3))  # Set figure size

    for ax, (start_time, stop_time) in zip(axs, times):
        # Calculate the index of the start and stop times
        start_index = int(start_time * sample_rate)
        stop_index = int(stop_time * sample_rate)

        # Ensure the stop time does not exceed the waveform length
        stop_index = min(stop_index, len(waveform))

        # Extract the desired segment of the waveform
        segment = waveform[start_index:stop_index]

        # Generate a time axis for the segment, starting at 0
        time_axis = np.linspace(start_time, stop_time, len(segment), endpoint=False)

        ax.plot(time_axis, segment)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
    fig.suptitle(title)

    return fig