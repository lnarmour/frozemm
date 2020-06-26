#!/bin/bash

make clean > /dev/null 2>&1
make > /dev/null 2>&1

Ns=(5 10 11 15);

for N in ${Ns[@]}; 
do
  err="$(./matmult01 $N | grep FAILED)";
  if [ -n "$err" ]; then
    echo "[FAILED]: ./matmult01 $N";
    exit 1
  fi
done
echo "[PASSED]";
