#!/bin/bash

N=4096;

for F in {0..1024..32};
do
  echo ./bin/transpose.fixedmem $N $F;
  ./bin/transpose.fixedmem $N $F;
  echo "";
  sleep 5;
done;

