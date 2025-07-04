# Helpful assistant or fruitful facilitator? Investigating how personas affect language model behavior
 
This repo holds source code for the paper "Helpful assistant or fruitful facilitator? Investigating how personas affect language model behavior".


## Requirement

- [miniforge](https://github.com/conda-forge/miniforge)

## Setting up 

1. Run the snippet below to install all dependencies:

```console
conda env create -f environment.yml
```
2. Download toxicity annotation data from https://maartensap.com/racial-bias-hatespeech/. Extract and place the "largeScale.csv" file in "./data/annWithAttitudes/"

## Persona generations
- Generations from all personas for all models and datasets is available in the "results" folder.


## Reproducing the experiments
- Notebook augment_attitude_questions.ipynb generates paraphrases for the attitude questionnaires.
- Notebook create_control_personas.ipynb generates the control personas.
- Script gen_preds.py obtains model predictions. The following example generates predictions for all datasets, persona, and control personas for the model Zephyr:
```console
python gen_preds.py "HuggingFaceH4/zephyr-7b-beta" --dataset all
python gen_preds.py "HuggingFaceH4/zephyr-7b-beta" --dataset all --control
```
- Script compute_results.py computes and saves results (hits, scores, and answer extraction) for all personas, models, and datasets.
- Notebook gen_graphics.ipynb reproduces analysis and figures in the paper.
- Notebook 'sig_testing.ipynb' runs the significance tests.


## Data notice
- Our data folder contains additional metadata from BBQ ( https://github.com/nyu-mll/, CC-BY-4.0 license) and MMLU (https://github.com/hendrycks/test, MIT license)
