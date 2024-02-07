#%%
import argparse
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio

import librosa
import librosa.display
import matplotlib.pyplot as plt
from IPython.display import Audio, display

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.distributions import Categorical
from tqdm.auto import tqdm
import sys
from s4_dx7.lib.visualistaion.audio import waveform_segment_figure
from s4_dx7.notebook.mel_spec_audio import plot_melspectrogram_and_play_button
sys.path.append('/home/athena/nintorac/s4_dx7/s4')
from src import utils
# from s4.src.dataloaders.audio import mu_law_decode
# from s4.src.models.baselines.wavenet import WaveNetModel
from train import SequenceLightningModule

#%%
@torch.inference_mode()
def generate(
    model,
    batch,
    tau=1.0,
    l_prefix=0,
    T=None,
    debug=False,
    top_p=1.0,
    benchmark=False,
    return_logprobs=False,
):

    x, _, *_ = batch # (B, L)
    x = x.to('cuda')
    T = x.shape[1] if T is None else T

    # # Special logic for WaveNet
    # if isinstance(model.model, WaveNetModel) and not benchmark:
    #     l_prefix += model.model.receptive_field
    #     T += model.model.receptive_field
    #     x = F.pad(x, (model.model.receptive_field, 0), value=128)

    # Set up the initial state
    model._reset_state(batch, device='cuda')

    # First sample
    x_t = x[:, 0]
    y_all = []
    logprobs = np.zeros(x.shape[0])
    entropy = np.zeros(x.shape[0])

    if debug:
        y_raw = []

    # Generation loop
    for t in tqdm(range(T)):

        # Step through the model with the current sample
        y_t = model.step(x_t)

        # Handle special loss functions such as ProjectedAdaptiveSoftmax
        if hasattr(model.loss, "compute_logits"): y_t = model.loss.compute_logits(y_t)

        if debug:
            y_raw.append(y_t.detach().cpu())

        # Output distribution
        probs = F.softmax(y_t, dim=-1)

        # Optional: nucleus sampling
        if top_p < 1.0:
            sorted_probs = probs.sort(dim=-1, descending=True)
            csum_probs = sorted_probs.values.cumsum(dim=-1) > top_p
            csum_probs[..., 1:] = csum_probs[..., :-1].clone()
            csum_probs[..., 0] = 0
            indices_to_remove = torch.zeros_like(csum_probs)
            indices_to_remove[torch.arange(sorted_probs.indices.shape[0])[:, None].repeat(1, sorted_probs.indices.shape[1]).flatten(), sorted_probs.indices.flatten()] = csum_probs.flatten()
            y_t = y_t + indices_to_remove.int() * (-1e20)

        # Sample from the distribution
        y_t = Categorical(logits=y_t/tau).sample()

        # Feed back to the model
        if t < l_prefix-1:
            x_t = x[:, t+1]
        else:
            x_t = y_t

            # Calculate the log-likelihood
            if return_logprobs:
                probs = probs.squeeze(1)
                if len(y_t.shape) > 1:
                    logprobs += torch.log(probs[torch.arange(probs.shape[0]), y_t.squeeze(1)]).cpu().numpy()
                else:
                    logprobs += torch.log(probs[torch.arange(probs.shape[0]), y_t]).cpu().numpy()
                entropy += -(probs * (probs + 1e-6).log()).sum(dim=-1).cpu().numpy()

        y_all.append(x_t.cpu())
        # y_all.append(y_t.cpu())

    y_all = torch.stack(y_all, dim=1) # (batch, length)

    if isinstance(model.model, WaveNetModel) and not benchmark:
        y_all = y_all[:, model.model.receptive_field:]


    if not return_logprobs:
        if debug:
            y_raw = torch.stack(y_raw)
            return y_all, y_raw
        return y_all
    else:
        assert not debug
        return y_all, logprobs, entropy


@hydra.main(config_path="configs", config_name="generate.yaml")
def main(config: OmegaConf):
    ### See configs/generate.yaml for descriptions of generation flags ###

    # Load train config from existing Hydra experiment
    if config.experiment_path is not None:
        config.experiment_path = hydra.utils.to_absolute_path(config.experiment_path)
        experiment_config = OmegaConf.load(os.path.join(config.experiment_path, '.hydra', 'config.yaml'))
        # config = OmegaConf.merge(config, experiment_config)
        config.model = experiment_config.model
        config.task = experiment_config.task
        config.encoder = experiment_config.encoder
        config.decoder = experiment_config.decoder
        config.dataset = experiment_config.dataset
        config.loader = experiment_config.loader

    # Special override flags
    if not config.load_data:
        OmegaConf.update(config, "train.disable_dataset", True)

    if config.n_batch is None:
        config.n_batch = config.n_samples
    OmegaConf.update(config, "loader.batch_size", config.n_batch)

    # Create the Lightning Module - same as train.py

    config = utils.train.process_config(config)
    utils.train.print_config(config, resolve=True)

    print("Loading model...")
    # assert torch.cuda.is_available(), 'Use a GPU for generation.'

    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)

    # Define checkpoint path smartly
    if not config.experiment_path:
        ckpt_path = hydra.utils.to_absolute_path(config.checkpoint_path)
    else:
        ckpt_path = os.path.join(config.experiment_path, config.checkpoint_path)
    print("Full checkpoint path:", ckpt_path)

    # Load model
    if ckpt_path.endswith('.ckpt'):
        model = SequenceLightningModule.load_from_checkpoint(ckpt_path, config=config, map_location='cpu')
        # model.to('cuda')
    elif ckpt_path.endswith('.pt'):
        model = SequenceLightningModule(config)
        # model.to('cuda')

        # Load checkpoint
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict)

    # Setup: required for S4 modules in SaShiMi
    for module in model.modules():
        if hasattr(module, '_setup_step'): module._setup_step()
    model.eval()

    if config.load_data:
        # Get the eval dataloaders
        eval_dataloaders = model.val_dataloader()
        dl = eval_dataloaders[0] if config.split == 'val' else eval_dataloaders[1]
    else:
        assert config.l_prefix == 0, 'Only unconditional generation when data is not loaded.'

    # Handle save directory intelligently
    if config.save_dir:
        save_dir = hydra.utils.to_absolute_path(config.save_dir)
    else:
        save_dir = os.path.join(os.getcwd(), "samples/")
    os.makedirs(save_dir, exist_ok=True)

    return model, dl

config_path = Path('/home/athena/nintorac/s4_dx7/s4/configs')
current_path = Path(os.getcwd())
rel_c_path = config_path.relative_to(current_path)
with initialize(version_base=None, config_path=rel_c_path.as_posix()):
    cfg = compose(
        config_name="generate.yaml",
        overrides=[
            f'experiment_path=/home/athena/nintorac/s4_dx7/experiments/s4-dx7-vc-fir-01',
            'checkpoint_path=checkpoints/val/loss.ckpt',
            '+experiment=audio/sashimi-sc09',
            '+dataset.limit=15',
            '+n_batches=1',
            'n_samples=3'


        ]
    )

    # +experiment=audio/sashimi-sc09 
    # loader.batch_size=14 
    # dataset=dx7 
    # +dataset.sr=8000 
    # +dataset.duration=2.5 
    # trainer.limit_train_batches=1000
    # trainer.limit_val_batches=100
    # trainer.accumulate_grad_batches=2 
    # +dataset.limit=20000 
    # +dataset.bit_rate=8 
    # +tolerance.id=42

model, loader = main(cfg)
batch = next(iter(loader))
with torch.no_grad():
    output = model(batch)

#%%



#%%
samples=0
print('source')
plot_melspectrogram_and_play_button(batch['x'][samples]/(2**(8))-1, sample_rate=8000)
print('target')
plot_melspectrogram_and_play_button(batch['y'][samples, ...,0]/(2**(8))-1, sample_rate=8000)
print('converted')
plot_melspectrogram_and_play_button(output[0][samples].max(-1)[1].cpu()/(2**(8))-1, sample_rate=8000)
plot_melspectrogram_and_play_button(
    output[0][samples].max(-1)[1].cpu()/(2**(8))-1-
    batch['y'][samples, ...,0]/(2**(8))-1
, sample_rate=8000)
# %%
waveform_segment_figure(batch['x'][samples]/(2**(8))-1,8000,[(0.,0.2), (0,0.02)])
waveform_segment_figure(batch['y'][samples]/(2**(8))-1,8000,[(0.,0.2), (0,0.02)])
waveform_segment_figure(output[0][samples].max(-1)[1].cpu()/(2**(8))-1,8000,[(0.,0.2), (0,0.02)])