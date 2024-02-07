#!/bin/bash
sudo apt-get install -y zsh pipx
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
pipx install poetry

~/s4/install.sh -b -p $HOME/miniconda

eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda init zsh
conda create python=3.11.2 -n=s -y
conda activate s
/home/ubuntu/.local/bin/poetry install
cd /home/ubuntu/s4/s4_dx7/s4/extensions/kernels
python setup.py install