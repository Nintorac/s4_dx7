from typing import List
import torch
from bitarray import bitarray
from torchaudio.transforms import Fade

NIBBLE_BITS = 4

def fsk_encode_bytes_batch(
    encode_bytes: List[bytes],
    sampling_rate = 44100,
    baud_rate = 300,
):
    """
    Generates the frequency codebook automatically based on the sampling rate
    
    taken from https://dsp.stackexchange.com/questions/80768/fsk-modulation-with-python
    """
    # TODO:   the author suggests some optimisations around the phi calcs
    #         allow for mfsk
    
    nyquist_limit = sampling_rate//2
    max_min_offset = (nyquist_limit)//10
    codebook = torch.linspace(max_min_offset, nyquist_limit-max_min_offset, 2)
    samples_per_bit = int(1.0 / baud_rate * sampling_rate)
    # tones representing bits, dummy data (0,1)
    arrs = []
    for item in encode_bytes:
        bits = bitarray(endian='big')
        bits.frombytes(item)
        bits_in_tones = [codebook[int(bit)] for bit in bits]
        arrs.append(torch.tensor(bits_in_tones))
    bit_arr = torch.stack(arrs)
    symbols_freqs = bit_arr.repeat_interleave(samples_per_bit, -1)


    # New lines here demonstrating continuous phase FSK (CPFSK)
    delta_phi = symbols_freqs * torch.pi / (sampling_rate / 2.0)
    phi = torch.cumsum(delta_phi, dim=1)
    signal = torch.sin(phi)
    return signal

def fsk_encode_syx(
    syx: List[bytearray],
    sampling_rate=44100,
    baud_rate=300
) -> torch.Tensor:
    """
    Encodes a syx message into an FSK signal
    adds some padding and a fade out to the end to avoid signal discontinuities
    """
    fade = Fade(fade_out_len=int(NIBBLE_BITS/baud_rate*sampling_rate)) # fade the final nibble of the padded encoding
    pad_bytes = b'0'
    padded_bytes = [signal + pad_bytes for signal in syx]
    signal = fsk_encode_bytes_batch(padded_bytes, sampling_rate, baud_rate)

    signal = fade(signal)
    return signal

