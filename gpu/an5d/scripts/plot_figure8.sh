#!/bin/bash

if [[ -z "$1" ]]; then
  echo usage: $0 GPU_NAME;
  exit 1;
fi
gpu=$1;

if [[ -z "$T" ]]; then
  T=1000;
fi
if [[ -z "$S" ]]; then
  S=16384;
fi

tmp_file="/tmp/.tmp_${gpu}_data_for_python.csv";
T=$T S=$S CSV=1 ./scripts/parse_figure8.sh $gpu > $tmp_file;

export TITLE="$gpu, S=$S, T=$T";
export GPU=$gpu;
export DATA="$tmp_file";
python scripts/matplotlib_figure8.py
