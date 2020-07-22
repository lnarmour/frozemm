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

mkdir -p $gpu;
for i in {1..16}; 
do
  binary="star2d3r-512-$i-512";
  if [[ ! -f "./bin/$binary" ]]; then
    continue;
  fi
  cmd="./bin/$binary -n 1 -s $S -t $T"; 
  echo $cmd;
  eval $cmd > $gpu/$binary.s$S.t$T.log
done;
