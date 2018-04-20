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
mkdir rl
virtualenv rl
git clone https://github.com/rcln/dospordos.git
source bin/activate

# Install dependencies
pip install numpy
pip install editdistance
pip install -U spacy
pip install sklearn
pip install nltk
pip install hdf5

# Configure for GPU


