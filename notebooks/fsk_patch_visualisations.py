#%%
from itertools import chain

from bitarray import bitarray
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
sampling_rate = 40000
baud_rate = 8000
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
dt = int(1/baud_rate*sampling_rate)  # samples per bit
truncate = dt * 32
skip = dt * 64
#%%
bits = bitarray(endian='big')
bits.frombytes(voices.bytes[0])
bytes_as_bits = chain.from_iterable(map(int, list(format(bin(25), '010')[2:])) for byte in voices.bytes[0])
bits_arr = np.array(list(bits)).repeat(dt)
# bits_arr = np.array(list(bytes_as_bits)).repeat(dt)
bits_arr_w = np.stack([bits_arr]* 500)
plt.imshow(bits_arr_w[skip:skip+truncate])
# %%

fig, (signal_ax, st_ax, spec_ax) = plt.subplots((3), sharex=True)

t = np.arange(0, signal.shape[1] / sampling_rate, 1.0 / sampling_rate)[skip:skip+truncate]
x_locations = np.arange(skip, skip+truncate, dt)/sampling_rate

for x in x_locations:
    signal_ax.axvline(x=x, color='k', linestyle='--')
    st_ax.axvline(x=x, color='k', linestyle='--')
    spec_ax.axvline(x=x, color='k', linestyle='--')
st_ax.plot(t, signal[0, skip:skip+truncate].numpy())
# st_ax.plot(t, signal[1, skip:skip+truncate].numpy()-1.5)
signal_ax.plot(t, bits_arr[skip:skip+truncate]-0.5)

f, t_s, Zxx = s.stft(signal[0], nfft=128, fs=sampling_rate, nperseg=4, noverlap=3, padded=True)
valid_ts = (t_s<t.max()) & (t_s>=t.min())
spec_ax.pcolormesh(t_s[valid_ts], f, np.abs(Zxx[...,valid_ts]))
# spec_ax.pcolormesh(t_s, f, np.abs(Zxx))
# %%
