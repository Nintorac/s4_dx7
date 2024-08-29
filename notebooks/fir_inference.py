# %%
from functools import partial, wraps
import json
from tempfile import TemporaryFile
import uuid
import numpy as np
import pyarrow as pa
from collections import defaultdict
import duckdb
from IPython.core.display import HTML, display
from matplotlib import pyplot as plt
import torch
from s4_dx7.lib.data.audio_data_module import AudioDataModule
from s4_dx7.lib.render import render_batch, to_midi
from s4_dx7.lib.s4.generate import load_experiment
from s4_dx7.lib.visualistaion.audio import create_melspec_figure, waveform_segment_figure, render_piano_roll, render_voice
import networkx as nx
from s4_dx7.lib.visualistaion.graph_transform_representation import draw_transform_graph
from s4_dx7.lib.visualistaion.utils import fig_to_png_data_uri, float_to_ogg_data_uri


icons = {
    'wave': 'https://upload.wikimedia.org/wikipedia/commons/f/f1/Sine_Wave_%28PSF%29.png',
    'figure': 'https://matplotlib.org/stable/_static/favicon.ico',
    'midi': 'https://upload.wikimedia.org/wikipedia/commons/c/c0/MIDI_LOGO.png'

}

#%%


#%%
model_path = '/home/ubuntu/s4/s4_dx7/s4/outputs/2024-08-27'
con = duckdb.connect('data/audio.db')
model, _, config = load_experiment(
        '05-49-27-731431',
        f'checkpoints/val/loss.ckpt',
        'audio/sashimi-dx7-vc-fir',
        load_data=False,
        experiment_root=f"{model_path}" 
    )
model = model.cuda()
source_patch, target_patch = AudioDataModule.get_voices()
# %%
import base64
from mido import MidiFile, MidiTrack, Message
import io

def midi_messages_to_base64(messages):
    TICKS_PER_BEAT = 1000
    QUARTERS_PER_BEAT = 4
    print(messages)
    messages = to_midi(messages, bpm=60)
    # Create an empty MIDI file and track
    midi_file = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    midi_file.tracks.append(track)
    last_time = 0
    # Add supplied messages to the track
    for msg in sorted(messages, key=lambda x: x.time):
        print('--')
        print(msg.time)
        last_time, msg.time = msg.time, int((msg.time-last_time) * TICKS_PER_BEAT * QUARTERS_PER_BEAT)
        print(msg.time)
        track.append(msg)
    
    # Save MIDI file to a BytesIO buffer instead of a file
    buf = io.BytesIO()
    midi_file.save(file=buf)
    midi_file.save('/home/athena/example.mid')
    buf.seek(0)  # Seek back to the start of the BytesIO buffer
    
    # Convert the MIDI file in the buffer to a base64 encoded string
    base64_midi = base64.b64encode(buf.read()).decode('utf-8')
    
    return 'data:audio/midi;base64,' + base64_midi

def normalize_signal(signal, bit_rate):
    return signal / 2 ** (bit_rate - 1) - 1
def greedy_decode(signal):
    return signal.argmax(-1)
def corrupt_signal(signal):
    return AudioDataModule.clean_signal(signal, 7, 8)
    return torch.clamp(signal + int(signal.float().mean()), 0, 2**(8-1))
    # return torch.clamp(signal, 0, 2**8-1)
# Function to create Mel Spectrogram plot
def create_mel_spectrogram_plot(signal, sample_rate, title):
    return create_melspec_figure(signal.squeeze(0), sample_rate=sample_rate, title=title)
# Function to plot waveform segment
def create_signal_segment_plot(signal, sample_rate, times, title):
    return waveform_segment_figure(signal.squeeze(), sample_rate, times, title=title)

def s4_dx7_vc_fir_00(*args, **kwargs):
    return model(*args, **kwargs)
audio_f = lambda signal: f"""
 <audio controls>
  <source src="{float_to_ogg_data_uri(signal.squeeze().numpy(), config.dataset.sr)}" type="audio/ogg">
  <a href="this tag included as a hack to get pyvis to render html"/>
Your browser does not support the audio element.
</audio> 
""".replace('\n', '') + '\n'
def html_encode_figure(fig):
    with TemporaryFile('wb+') as f:
        fig.savefig(f)
        f.seek(0)
        png_bytes = f.read()

    encoded = base64.b64encode(png_bytes).decode('utf-8')
    return '<img src=\'data:image/png;base64,{}\'>\n'.format(encoded)
#%%
query = con.query('select * from melodies limit 1 offset 29')
df = query.to_df()
ar = query.to_arrow_table()

target_signal = render_voice(ar['notes'], target_patch, config.dataset)
source_signal = render_voice(ar['notes'], source_patch, config.dataset)
source_signal = AudioDataModule.clean_signal(source_signal, 8, 8)[..., 1:]
target_signal = AudioDataModule.clean_signal(target_signal, 8, 8)[..., 1:]

with torch.no_grad():
    generated_signal_logits = s4_dx7_vc_fir_00((source_signal.cuda(), target_signal.cuda().unsqueeze(-1)))[0]

generated_signal = greedy_decode(generated_signal_logits.cpu())

# Assuming render_voice, model, create_melspec_plot, plot_signal_pltoeform_segment are defined elsewhere

# Generate signals
signals = target_signal, source_signal, generated_signal
normalize_signal_w_rate = partial(normalize_signal, bit_rate=config.dataset.bit_rate)
(normalised_target_signal,
  normalised_source_signal,
        generated_signal, 
        ) = map(normalize_signal_w_rate, signals)

#%%
# Creating Mel Spectrogram Plots

# Plot waveform segments
times = [(0,2),(0,0.2),(0,0.02)]


# target_te1_signal_plot = create_signal_segment_plot(normalised_target_signal, config.dataset.sr, 0., .2, "Target Signal - 200ms Segment")
# target_te2_signal_plot = create_signal_segment_plot(normalised_target_signal, config.dataset.sr, 0., .02, "Target Signal - 20ms Segment")
target_signal_plot = create_signal_segment_plot(normalised_target_signal, config.dataset.sr, times, "Target Signal Segments @ 2000ms, 200ms, 20ms")
target_signal_mel = create_mel_spectrogram_plot(normalised_target_signal, config.dataset.sr, 'Target Signal Mel Spectrogram')
target_audio = audio_f(normalised_target_signal).replace('\n', '') + '\n'
print((target_audio + html_encode_figure(target_signal_plot) + html_encode_figure(target_signal_mel)).strip())

#%%
source_signal_plot = create_signal_segment_plot(normalised_source_signal, config.dataset.sr, times, "Source Signal Segments @ 2000ms, 200ms, 20ms")
source_signal_mel = create_mel_spectrogram_plot(normalised_source_signal, config.dataset.sr, 'Source Signal Mel Spectrogram')
source_audio = audio_f(normalised_source_signal)
print(source_audio + html_encode_figure(source_signal_plot) +'\n' + html_encode_figure(source_signal_mel))
#%%
generated_signal_plot = create_signal_segment_plot(generated_signal, config.dataset.sr, times, "Generated Signal Segments @ 2000ms, 200ms, 20ms")
generated_signal_mel = create_mel_spectrogram_plot(generated_signal, config.dataset.sr, 'Generated Signal Mel Spectrogram')
generated_audio = audio_f(generated_signal)
print(generated_audio + html_encode_figure(generated_signal_plot) + html_encode_figure(generated_signal_mel))
# %%
