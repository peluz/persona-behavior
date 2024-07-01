#!/bin/bash

mamba env create -f environment.yml
mamba activate persona-biases
python -m ipykernel install --user --name persona-biases