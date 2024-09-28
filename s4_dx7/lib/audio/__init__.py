import numpy
import torch


def float_to_pcm16(audio, bit_rate=16):

    ints = ((audio + 1) * (2**(bit_rate-1))).astype(numpy.int32)

    return ints


def pcm16_to_float(pcm16: torch.Tensor, bit_rate=16) -> torch.Tensor:
    return (pcm16.float() / (2**(bit_rate-1)))-1