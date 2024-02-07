import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import numpy as np

from s4_dx7.lib.visualistaion.audio import create_play_button, create_melspec_figure

# Function that combines plotting of the mel spectrogram and displaying the play button
def plot_melspectrogram_and_play_button(audio_signal, sample_rate=22050):
    
    # Display play button
    audio = create_play_button(audio_signal, sample_rate)

    # Plot the Mel Spectrogram
    plot = create_melspec_figure(audio_signal, sample_rate)
    
    display(plot, audio)
