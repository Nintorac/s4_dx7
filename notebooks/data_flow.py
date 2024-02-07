# %%
from functools import partial, wraps
import json
import uuid
import numpy as np
import pyarrow as pa
from collections import defaultdict
import duckdb
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

# Define the context manager for function execution tracking
class FunctionDagContext:
    def __init__(self):
        self.DAG = nx.DiGraph()
        self.current_node = None
        self.latest_id = 0

    def __enter__(self):
        # Nothing to return upon entering the context
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup or resources release if needed when context exits
        pass

    def dag_wrapped_function(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(self, 'DAG'):
                self.DAG = nx.DiGraph()
            # Generate an identifier for the input
            if isinstance(args[0], uuid.UUID):
                input_identifier, *args = args
                self.DAG.add_node(input_identifier)
                # Add input as a node to the DAG
            else:
                input_identifier = uuid.uuid4()
            input_node = (input_identifier, args, kwargs)
            
            # Execute the Function
            result = func(*args, **kwargs)
            
            # Generate an identifier for the output
            output_identifier = uuid.uuid4()
            # Create output node
            output_node = (output_identifier, result)

            # Add output as a node to DAG
            self.DAG.add_node(output_identifier)
            
            # Add edge representing the function from input node to output node
            self.DAG.add_edge(input_identifier, output_identifier, function=func.__name__)
            
            # Return result with identifier (as tuple to make it a single return value)
            return output_node
        return wrapper    
    
    # def dag_wrapped_function(self, func):

    #     node_map = {}
    #     @wraps(func)
    #     def wrapper(*args, **kwargs):
    #         if len(args)==1:
    #             current_id = self.latest_id
    #             value = args
    #             self.latest_id += 1
    #         else:
    #             current_id, *value = args
    #         # Before Function Execution: Update DAG
    #         previous_node = node_map.get(current_id)
    #         self.current_node = func.__name__
    #         node_map[current_id] = self.current_node
    #         if previous_node is not None:
    #             self.DAG.add_edge(previous_node, self.current_node)
    #         else:
    #             # Add the node if it does not depend on a previous function
    #             self.DAG.add_node(self.current_node)
                
    #         # Execute the Function
    #         result = current_id, func(*value, **kwargs)
            
    #         return result
    #     return wrapper


#%%
from huggingface_hub import snapshot_download
model_path = snapshot_download(repo_id="nintorac/s4_dx7", allow_patterns="s4-dx7-vc-fir/s4-dx7-vc-fir-00/*")

con = duckdb.connect('data/audio.db')
query = con.query('select * from melodies limit 1 offset 6666')
df = query.to_df()
ar = query.to_arrow_table()
model, _, config = load_experiment(
        's4-dx7-vc-fir-00',
        f'checkpoints/val/loss.ckpt',
        'audio/sashimi-dx7-vc-fir',
        load_data=False,
        experiment_root=f"{model_path}/s4-dx7-vc-fir" 
    )
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

#%%
target_signal = render_voice(ar['notes'], target_patch, config.dataset)
source_signal = render_voice(ar['notes'], source_patch, config.dataset)
broken_source_signal = corrupt_signal(source_signal)[..., :-1] 
broken_target_signal = corrupt_signal(target_signal)[..., 1:] 
source_signal = AudioDataModule.clean_signal(source_signal, 8, 8)[..., :-1]
target_signal = AudioDataModule.clean_signal(target_signal, 8, 8)[..., 1:]

with torch.no_grad():
    generated_signal_logits = s4_dx7_vc_fir_00((source_signal, target_signal.unsqueeze(-1)))[0]
    broken_source_generated_signal_logits = s4_dx7_vc_fir_00((broken_source_signal, target_signal.unsqueeze(-1)))[0]

generated_signal_broken_source = greedy_decode(broken_source_generated_signal_logits)
generated_signal = greedy_decode(generated_signal_logits)

# Assuming render_voice, model, create_melspec_plot, plot_signal_pltoeform_segment are defined elsewhere

# Generate signals
signals = target_signal, source_signal, broken_source_signal, generated_signal, generated_signal_broken_source, broken_target_signal
normalize_signal_w_rate = partial(normalize_signal, bit_rate=config.dataset.bit_rate)
(normalised_target_signal,
  normalised_source_signal,
    normalised_broken_source_signal,
      normalised_broken_generated_signal,
        normalised_working_generated_signal, 
        normalised_broken_target_signal, 
        ) = map(normalize_signal_w_rate, signals)

# Creating Mel Spectrogram Plots
source_signal_mel = create_mel_spectrogram_plot(normalised_source_signal, config.dataset.sr, 'Source Signal Mel Spectrogram')
target_signal_mel = create_mel_spectrogram_plot(normalised_target_signal, config.dataset.sr, 'Target Signal Mel Spectrogram')
broken_source_signal_mel = create_mel_spectrogram_plot(normalised_source_signal, config.dataset.sr, 'Broken Source Signal Mel Spectrogram')
broken_target_signal_mel = create_mel_spectrogram_plot(normalised_broken_target_signal, config.dataset.sr, 'Broken Target Signal Mel Spectrogram')
generated_signal_mel = create_mel_spectrogram_plot(normalised_broken_generated_signal, config.dataset.sr, 'Generated Signal Mel Spectrogram')
working_generated_signal_mel = create_mel_spectrogram_plot(normalised_working_generated_signal, config.dataset.sr, 'Generated Signal from Broken Source Mel Spectrogram')

# Plot waveform segments
times = [(0,2),(0,0.2),(0,0.02)]


# target_te1_signal_plot = create_signal_segment_plot(normalised_target_signal, config.dataset.sr, 0., .2, "Target Signal - 200ms Segment")
# target_te2_signal_plot = create_signal_segment_plot(normalised_target_signal, config.dataset.sr, 0., .02, "Target Signal - 20ms Segment")
target_signal_plot = create_signal_segment_plot(normalised_target_signal, config.dataset.sr, times, "Target Signal Segments @ 2000ms, 200ms, 20ms")

# source_te1_signal_plot = create_signal_segment_plot(normalised_source_signal, config.dataset.sr, 0., .2, "Source Signal - 200ms Segment")
# source_te2_signal_plot = create_signal_segment_plot(normalised_source_signal, config.dataset.sr, 0., .02, "Source Signal - 20ms Segment")
# source_te0_signal_plot = create_signal_segment_plot(normalised_source_signal, config.dataset.sr, 0., 2, "Source Signal - 2000ms Segment")
source_signal_plot = create_signal_segment_plot(normalised_source_signal, config.dataset.sr, times, "Source Signal Segments @ 2000ms, 200ms, 20ms")

# generate_te1_signal_plot = create_signal_segment_plot(normalised_broken_generated_signal, config.dataset.sr, 0., .2, "Generated Signal - 200ms Segment")
# generate_te1_signal_plot = create_signal_segment_plot(normalised_broken_generated_signal, config.dataset.sr, 0., .2, "Generated Signal - 200ms Segment")
# generate_te1_signal_plot = create_signal_segment_plot(normalised_broken_generated_signal, config.dataset.sr, 0., 2, "Generated Signal - 2000ms Segment")
generate_signal_plot = create_signal_segment_plot(normalised_broken_generated_signal, config.dataset.sr, times, "Generated Signal Segments @ 2000ms, 200ms, 20ms")

# broken_te2_signal_plot = create_signal_segment_plot(normalised_broken_source_signal, config.dataset.sr, 0., .02, "Broken Source Signal - 20ms Segment")
# broken_te1_signal_plot = create_signal_segment_plot(normalised_broken_source_signal, config.dataset.sr, 0., .2, "Broken Source Signal - 200ms Segment")
# broken_te0_signal_plot = create_signal_segment_plot(normalised_broken_source_signal, config.dataset.sr, 0., 2, "Broken Source Signal - 2000ms Segment")
broken_source_signal_plot = create_signal_segment_plot(normalised_broken_source_signal, config.dataset.sr, times, "Broken Source Signal Segments @ 2000ms, 200ms, 20ms")
broken_target_signal_plot = create_signal_segment_plot(normalised_broken_target_signal, config.dataset.sr, times, "Broken Target Signal Segments @ 2000ms, 200ms, 20ms")

# working_generated_te1_signal_plot = create_signal_segment_plot(normalised_working_generated_signal, config.dataset.sr, 0., .2, "Generated Broken Source Signal - 200ms Segment")
# working_generated_te2_signal_plot = create_signal_segment_plot(normalised_working_generated_signal, config.dataset.sr, 0., .02, "Generated Broken Source Signal - 20ms Segment")
# working_generated_te0_signal_plot = create_signal_segment_plot(normalised_working_generated_signal, config.dataset.sr, 0., 2, "Generated Broken Source Signal - 2000ms Segment")
working_generated_signal_plot = create_signal_segment_plot(normalised_working_generated_signal, config.dataset.sr, times, "Working Generated Signal Segments @ 2000ms, 200ms, 20ms")
#%%
# G=nx.DiGraph()
# # G.add_node('json_midi')
# # json_midi = G.add_node('mido_midi')
# source_pcm_audio = G.add_node('source_pcm_audio')
# target_pcm_audio = G.add_node('target_pcm_audio')
# corrupt_pcm_audio = G.add_node('corrupt_pcm_audio')
# # source_generated_pcm_audio = G.add_node('source_generated_pcm_audio')
# # corrupt_generated_pcm_audio = G.add_node('corrupt_generated_pcm_audio')
# audio_uri = float_to_ogg_data_uri(normalised_target_signal.squeeze().numpy(), config.dataset.sr)
# G.add_edge('source_pcm_audio','corrupt_pcm_audio')
# audio = f"""
#  <audio controls>
#   <source src="{audio_uri}" type="audio/ogg">
#   <a href="this tag included as a hack to get pyvis to render html"/>
# Your browser does not support the audio element.
# </audio> 
# """
# G.nodes['source_pcm_audio'].update({'title': f'{audio}'})
# draw_transform_graph(G, image={
#     node: fig_to_png_data_uri(target_te2_signal_plot)
#    for node in nodes 
#     })

#%%
import networkx as nx
import matplotlib.pyplot as plt
img_f = lambda img: f"""
 <img src="{fig_to_png_data_uri(img)}">
  <a href="this tag included as a hack to get pyvis to render html"/>
</img> 
"""

audio_f = lambda signal: f"""
 <audio controls>
  <source src="{float_to_ogg_data_uri(signal.squeeze().numpy(), config.dataset.sr)}" type="audio/ogg">
  <a href="this tag included as a hack to get pyvis to render html"/>
Your browser does not support the audio element.
</audio> 
"""
midi_f = lambda phrase_messages: f"""
 <a href="{midi_messages_to_base64(phrase_messages)}" download="example.mid">Download MIDI file</a>
"""



# Create a graph
G = nx.DiGraph()

# Declare nodes
nodes = [
    'midi',
    'normalised_target_signal',
    'normalised_source_signal',
    'normalised_broken_source_signal',
    'normalised_broken_target_signal',
    'normalised_broken_generated_signal',
    'normalised_working_generated_signal',
    'source_signal_mel',
    'target_signal_mel',
    'broken_target_signal_mel',
    'broken_source_signal_mel',
    'generated_signal_mel',
    'working_generated_signal_mel',
    'target_signal_plot',
    'source_signal_plot',
    'generate_signal_plot',
    'broken_source_signal_plot',
    'piano_roll_plot'
]

# Declare edges with labels
edges = [
    ('normalised_broken_generated_signal', 'create_mel_spectrogram_plot', 'generated_signal_mel'),
    ('normalised_broken_generated_signal', 'create_signal_segment_plot', 'generate_signal_plot'),
    ('normalised_broken_source_signal', 'create_mel_spectrogram_plot', 'broken_source_signal_mel'),
    ('normalised_broken_source_signal', 'create_signal_segment_plot', 'broken_source_signal_plot'),
    ('normalised_broken_source_signal', 's4_dx7_vc_fir_00', 'normalised_working_generated_signal'),
    ('normalised_broken_target_signal', 'create_mel_spectrogram_plot', 'broken_target_signal_mel'),
    ('normalised_broken_target_signal', 'create_mel_spectrogram_plot', 'broken_target_signal_plot'),
    ('normalised_broken_target_signal', 'loss', 'normalised_working_generated_signal'),
    ('normalised_source_signal', 'create_mel_spectrogram_plot', 'source_signal_mel'),
    ('normalised_source_signal', 's4_dx7_vc_fir_00', 'normalised_broken_generated_signal'),
    ('normalised_source_signal', 's4_dx7_vc_fir_01 - maybe?', 'normalised_target_signal'),
    ('normalised_source_signal','corrupt', 'normalised_broken_source_signal'),
    ('normalised_source_signal','create_signal_segment_plot', 'source_signal_plot'),
    ('normalised_target_signal', 'corrupt', 'normalised_broken_target_signal'),
    ('normalised_target_signal', 'create_mel_spectrogram_plot', 'target_signal_mel'),
    ('normalised_target_signal', 'create_signal_segment_plot', 'target_signal_plot'),
    ('normalised_working_generated_signal', 'create_mel_spectrogram_plot', 'working_generated_signal_mel'),
    ('normalised_working_generated_signal', 'create_signal_segment_plot', 'working_generated_signal_plot'),
    ('midi', 'render', 'normalised_target_signal'),
    ('midi', 'render', 'normalised_source_signal'),
    ('midi', 'render_piano_roll', 'piano_roll_plot')
    ]

pretty_names = {
    'midi': 'MIDI',
    'normalised_target_signal': 'Target Signal',
    'normalised_source_signal': 'Source Signal',
    'normalised_broken_source_signal': 'Corrupted Source Signal',
    'normalised_broken_target_signal': 'Corrupted Target signal',
    'normalised_broken_generated_signal': 'Clean Source Generated Signal ',
    'normalised_working_generated_signal': 'Corrupt Source Generated Signal',
    'source_signal_mel': 'Source Signal Mel-Spec',
    'target_signal_mel': 'Target Signal Mel-Spec',
    'broken_target_signal_mel': 'Corrupt Target Signal Mel-Spec',
    'broken_source_signal_mel': 'Corrupt Source Signal Mel-Spec',
    'generated_signal_mel': 'Clean Source Generated Signal Mel-Spec',
    'working_generated_signal_mel': 'Corrupt Source Generated Signal Mel-Spec',
    'target_signal_plot': 'Target Signal Waveform',
    'source_signal_plot': 'Source Signal Waveform',
    'generate_signal_plot': 'Clean Source Generated Signal Waveform',
    'broken_source_signal_plot': 'Corrupt Source Generated Signal Waveform',
    'piano_roll_plot': 'Piano Roll',
 }
# Add nodes to the graph
G.add_nodes_from(nodes)

# Add edges to the graph with labels
for edge in edges:
    G.add_edge(edge[0], edge[2], label=edge[1], font={'size': 8})


# Draw the graph
pos = nx.layout.spring_layout(G)
# pos = nx.layout.kamada_kawai_layout(G)
labels = {node: node for node in G.nodes}
nx.draw_networkx(G, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=80, edge_color='black', font_size=1)

node_title = {
    'normalised_target_signal': audio_f(normalised_target_signal),
    'normalised_source_signal': audio_f(normalised_source_signal),
    'normalised_broken_source_signal': audio_f(normalised_broken_source_signal),
    'normalised_broken_target_signal': audio_f(normalised_broken_target_signal),
    'normalised_broken_generated_signal': audio_f(normalised_broken_generated_signal),
    'normalised_working_generated_signal': audio_f(normalised_working_generated_signal),
    'source_signal_mel': img_f(source_signal_mel),
    'target_signal_mel': img_f(target_signal_mel),
    'broken_target_signal_mel': img_f(broken_target_signal_mel),
    'broken_source_signal_mel': img_f(broken_source_signal_mel),
    'generated_signal_mel': img_f(generated_signal_mel),
    'working_generated_signal_mel': img_f(working_generated_signal_mel),
    'target_signal_plot': img_f(target_signal_plot),
    'source_signal_plot': img_f(source_signal_plot),
    'generate_signal_plot': img_f(generate_signal_plot),
    'broken_source_signal_plot': img_f(broken_source_signal_plot),
    'broken_target_signal_plot': img_f(broken_target_signal_plot),
    'working_generated_signal_plot': img_f(working_generated_signal_plot),
    'midi': midi_f(df.notes[0]),
    'piano_roll_plot': img_f(render_piano_roll(to_midi(df.notes[0])))
    
}
node_type = {
    'normalised_target_signal': 'wave',
    'normalised_source_signal': 'wave',
    'normalised_broken_source_signal': 'wave',
    'normalised_broken_target_signal': 'wave',
    'normalised_working_generated_signal': 'wave',
    'normalised_broken_generated_signal': 'wave',
    'source_signal_mel': 'figure',
    'target_signal_mel': 'figure',
    'broken_target_signal_mel': 'figure',
    'broken_source_signal_mel': 'figure',
    'generated_signal_mel': 'figure',
    'working_generated_signal_mel': 'figure',
    'target_signal_plot': 'figure',
    'source_signal_plot': 'figure',
    'generate_signal_plot': 'figure',
    'broken_target_signal_plot': 'figure',
    'broken_source_signal_plot': 'figure',
    'working_generated_signal_plot': 'figure',
    'piano_roll_plot': 'figure',
    'midi': 'midi'
}
for node, label in pretty_names.items():
    G.nodes[node].update({'label': label})


for node, node_type in node_type.items():
    G.nodes[node].update({

    # 'image': 'https://www.w3schools.com/w3css/img_lights.jpg',
    'image': icons[node_type],
    'shape': 'image',
    'size': 19 if node_type is not 'figure' else 5,
    'font': {'size': 19 if node_type is not 'figure' else 5}
})
for node, title in node_title.items():
    G.nodes[node].update({'title': node_title.get(node)})
draw_transform_graph(G)


# %%
