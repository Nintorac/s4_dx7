# s4_dx7

To train the s4-dx7-vc FIR model simply run the following command, and wait for a few days.The configuration is tuned for an A10 GPU. 16 cores and 32gb of memory should do it? The lambda instance was overkill for the processing requirements.

## Running the model

### Inference

The trained weights are available on hugging face, check out noteboo found here 'notebooks/data_flow.py' which contains an example running inference. This will happily run on a CPU-only machine relatively quickly for small batch sizes. 

### Training

```bash
cd s4
python -m train +experiment=audio/sashimi-dx7-vc-fir
```

This will begin training though it's not recommended since there a several known bugs. see the blog

## Blog

Can be found here


## Install

A poetry lock is provided to help ensure dependencies. Simply run the following from the root of the project.


Make sure to [install poetry](https://python-poetry.org/docs/) in a different Python environment to the one you're using to train.
```bash
poetry install
```