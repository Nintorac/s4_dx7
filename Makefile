
.PHONY: install_data_deps install_train_deps


install_data_deps:
	poetry install --only vis --only data


