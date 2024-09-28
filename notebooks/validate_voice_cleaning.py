#%% [markdown]
# you should run dexed alongside this notebook,
# it will send out dx7 patches to the midi bus 
# if dexed is running it should pick it up and load the patch
# play around with it and load the next patch by providing empty input
# if you want to stop type anything into the input
#%%
import json
from pathlib import Path
import duckdb
import mido
from s4_dx7.lib.dx7 import DX7Single, clean_voice
from s4_dx7.utils import get_duckdb_connection

#%%
duck = get_duckdb_connection('data/dev.duckdb')
# duck.execute("attach 'data/audio.db' as audio")
#%%
# voices = duck.query('select * from dx7_voices')
voices = duck.query('select * from dx7_voices').limit(100)
print(mido.get_input_names())
## change if input name is different
port = mido.open_output('Midi Through:Midi Through Port-0 14:0')
for voice in voices.to_df().itertuples():
    print(voice.path)
    print(voice.index)
    port.send(mido.Message.from_bytes(voice.bytes))
    i =  input()
    if i: break
#%%
voices = duck.query("select * from metadata_dx7_voices order by random() limit 100")
voices
# %%
mido.get_input_names()
port = mido.open_output('Midi Through:Midi Through Port-0 14:0')
for voice in voices.to_df().itertuples():
    print(voice.voice_id)
    print(voice.metadata)
    port.send(mido.Message.from_bytes(voice.bytes))
    i =  input()
    if i: break
# %%
# %%
