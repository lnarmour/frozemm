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

for bt in {1..16}; 
do
  filename="${stencil}-${bs}-${bt}-${sl}.s$S.t$T.log";
  if [[ ! -f "./$gpu/$filename" ]]; then
    continue;
  fi

  if [[ -n $CSV ]]; then
    line="$(cat ./$gpu/$filename | grep Average | sed "s~Average:[ ]*~~")";
    line="$(echo $line | sed 's~, ~,~g')";
    gflops=`echo $line | cut -d ',' -f 1 | cut -d ' ' -f 1`
    ms=`echo $line | cut -d ',' -f 2 | cut -d ' ' -f 1`
    joules=`echo $line | cut -d ',' -f 3 | cut -d ' ' -f 1`
    echo "$bt,$gflops,$ms,$joules";
  else
    cat ./$gpu/$filename | grep Average | sed "s~Average:\(.*\)~$(if [[ "$bt" -lt 10 ]]; then printf ' '; fi;)$bt\1~";
  fi
done;
