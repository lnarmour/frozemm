#!/bin/bash

if [[ -z "$1" || -z "$2" ]]; then
  echo usage: $0 GPU_NAME STENCIL;
  exit 1;
fi
gpu=$1;
stencil=$2;

bs=512;
sl=512;

if [[ -z "$T" ]]; then
  T=1000;
fi
if [[ -z "$S" ]]; then
  S=16384;
fi


mkdir -p $gpu;
for i in {1..16}; 
do
  binary="${stencil}-${bs}-${i}-${sl}";
  log_file="$gpu/$binary.s$S.t$T.log";
  cmd="./bin/$binary -n 1 -s $S -t $T"; 
  printf "running: $cmd ... ";
  if [[ -f "$log_file" && -z $FORCE ]]; then
    printf "log already exists, skipping.\n";
    continue;
  fi
  if [[ ! -f "./bin/$binary" ]]; then
    printf "binary doesn't exist, skipping.\n";
    continue;
  fi
  eval $cmd > $log_file;
  printf "done\n";
done;
