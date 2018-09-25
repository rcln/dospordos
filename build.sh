#!/usr/bin/env bash

:'
numpy,
keras with gpu support,
nltk,
spacy,
editdistance,
sklearn, hdf5 or cuDNN
'

# Make virtual env
virtualenv -p python3 venv-dospordos
git clone https://github.com/rcln/dospordos.git
source venv-dospordos/bin/activate

# Install dependencies
pip install numpy
pip install editdistance
pip install -U spacy
# python -m spacy download en
pip install sklearn
pip install nltk
pip install h5py

# Configure for GPU


