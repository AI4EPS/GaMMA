#!/bin/bash

set -e
set -x

ANACONDA_ORG="zhuwq0"
ANACONDA_TOKEN="$ANACONDA_UPLOAD_TOKEN"

# Install Python 3.8 because of a bug with Python 3.9
export PATH=$CONDA/bin:$PATH
conda create -n upload -y python=3.8
source activate upload
conda install -y anaconda-client

# Force a replacement if the remote file already exists
anaconda -t $ANACONDA_TOKEN upload --force -u $ANACONDA_ORG dist/artifact/*
echo "Index: https://pypi.anaconda.org/$ANACONDA_ORG/simple"
