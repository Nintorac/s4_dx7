
.PHONY: install_data_deps install_train_deps


install_data_deps:
	poetry install --only vis --only data


render_multivoice_notebook:
	jupytext --to ipynb  notebooks/dataloaders/multi_voice2voice.py
	jupyter nbconvert --to markdown --execute notebooks/dataloaders/multi_voice2voice.ipynb --no-input --output-dir='./docs/experiment_logs/dataset_visualisations'
