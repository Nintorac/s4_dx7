#%%
from s4_dx7.lib.audio.digital import fsk_encode_bytes_batch 
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as s
from matplotlib import pyplot as plt
from s4_dx7.lib.audio.digital import fsk_encode_syx

from s4_dx7.utils import get_duckdb_connection
duck = get_duckdb_connection('data/dev.duckdb')
#%%
signal = fsk_encode_bytes_batch([b'lB', b'ab'], sampling_rate=8000)
sampling_rate = 16000
baud_rate = sampling_rate//8
t = np.arange(0, signal.shape[1] / sampling_rate, 1.0 / sampling_rate)
plt.plot(t, signal[0].numpy()+1)
plt.plot(t, signal[1].numpy()-1)
signal = fsk_encode_syx([b'ab', b'cd'], sampling_rate=sampling_rate)
t = np.arange(0, signal.shape[1] / sampling_rate, 1.0 / sampling_rate)
plt.plot(t, signal[0].numpy()+1)
plt.plot(t, signal[1].numpy()-1)
# %%
voices = duck.query("select * from metadata_dx7_voices order by random() limit 2").to_df()
signal = fsk_encode_syx(voices.bytes, sampling_rate=sampling_rate, baud_rate=baud_rate)
print(signal.shape)
#%%
truncate = 200
skip = 500
t = np.arange(0, signal.shape[1] / sampling_rate, 1.0 / sampling_rate)[skip:skip+truncate]
dt = 1/baud_rate*sampling_rate  # your desired spacing
x_locations = np.arange(skip, skip+truncate, dt)/sampling_rate

for x in x_locations:
    plt.axvline(x=x, color='k', linestyle='--')

plt.plot(t, signal[0, skip:skip+truncate].numpy()+1)
plt.plot(t, signal[1, skip:skip+truncate].numpy()-1)
#%%

for x in x_locations:
    plt.axvline(x=x, color='k', linestyle='--')
f, t, Zxx = s.stft(signal[0, skip:skip+truncate], nfft=128, fs=sampling_rate, nperseg=dt//2, noverlap=dt//4, padded=True)
plt.pcolormesh(t+skip/sampling_rate, f, np.abs(Zxx))
# %%

