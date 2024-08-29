from collections import defaultdict
from typing import Optional, Mapping, Tuple, Union
import logging
from functools import partial
import math
import numpy as np
from scipy import special as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from einops import rearrange, repeat
# from s4.models.s4.s4 import Activation, DropoutNd, FFTConv, LinearActivation, get_logger

log = logging #get_logger(__name__)

from mamba_ssm import Mamba2 as Mamba


class Mamba2(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=64,
        headdim=8,
        A_init_range=(1.,1.1),
        **kwargs
    ):
        super().__init__()

        self.layer = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=d_model, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            A_init_range=A_init_range,
            headdim=headdim # https://github.com/state-spaces/mamba/issues/351#issuecomment-2167091940
            # expand=6,
        ).to("cuda")

    def forward(self, x, lengths=None, **kwargs): # absorbs return_output and transformer src mask
        """
        x: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as x
        """
        # raise ValueError(x.shape)
        # logging.critical(x.shape)
        x = self.layer(x.transpose(2,1)).transpose(2,1)
        # logging.critical(x.shape)
        return x, None
    
    def step(self, x, state):
        """Step one time step as a recurrent model. Intended to be used during validation.

        x: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """

        raise NotImplementedError()

    def default_state(self, *batch_shape, device=None):
        raise NotImplementedError()
        return self.layer.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.layer.d_state

    @property
    def d_output(self):
        return self.layer.d_model

    @property
    def state_to_tensor(self):

        raise NotImplementedError()
        return self.layer.state_to_tensor
