#!/bin/bash

set -e
set -x

ANACONDA_ORG="zhuwq0"
ANACONDA_TOKEN="$ANACONDA_UPLOAD_TOKEN"

pip install git+https://github.com/Anaconda-Server/anaconda-client

# Force a replacement if the remote file already exists
anaconda -t $ANACONDA_TOKEN upload --force -u $ANACONDA_ORG dist/artifact/*
echo "Index: https://pypi.anaconda.org/$ANACONDA_ORG/simple"
