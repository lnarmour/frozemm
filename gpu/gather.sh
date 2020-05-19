#!/bin/bash

nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory --format=csv,nounits --loop-ms=100 > ttss.15k.smi.log &
sleep 1;
pid="$(ps aux | grep nvidia-smi | grep query| cut -f 2 -d ' ')";

N=15000
PIJ=(256 512 1024 2048 4096)

i=1;
for pij in ${PIJ[@]};
do
  eval "TK=({$pij..64..-64})"
  for tk in ${TK[@]}; 
  do 
    echo "$(nvidia-smi --query-gpu=timestamp --format=csv,noheader), $N, $pij, $pij, $tk, $(./MM $N $pij $pij $tk)"; 
  done;
  i=$(($i+1))
done;

kill $pid;

