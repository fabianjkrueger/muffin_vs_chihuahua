#!/bin/bash

# Assemble paths so script can be executed independent of working directory

# directory where script is located, regardless of where it's called from
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# repo root (parent of scripts directory)
REPO_ROOT="$( cd "$SCRIPT_DIR" && cd .. && pwd )"
# data directory
DATA_DIR="$REPO_ROOT/data"
# raw data directory
DATA_RAW_DIR="$DATA_DIR/raw"

# make sure directories exist
mkdir -p "$DATA_DIR"
mkdir -p "$DATA_RAW_DIR"

# print start of download message
echo "Downloading data..."

# download data
curl -L -o "$DATA_RAW_DIR/muffin-vs-chihuahua-image-classification.zip"\
  https://www.kaggle.com/api/v1/datasets/download/samuelcortinhas/muffin-vs-chihuahua-image-classification

# unzip data
unzip "$DATA_RAW_DIR/muffin-vs-chihuahua-image-classification.zip" -d "$DATA_RAW_DIR"

# remove zip file
rm "$DATA_RAW_DIR/muffin-vs-chihuahua-image-classification.zip"

# print success message
echo "Data downloaded and unzipped successfully"
