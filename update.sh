#!/bin/bash
mamba activate persona-biases
mamba env update --prune -f environment.yml
python -m ipykernel install --user --name persona-biases


