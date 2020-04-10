#!/bin/bash

rm -rf results.log;

N=5000
Ps=({100..3000..100})
TKs=({1000..25..-25})

count $N $N $N $N >> results.log; 

for PI in ${Ps[@]}; 
do 
  for PJ in ${Ps[@]}; 
  do
    for TK in ${TKs[@]}; 
    do
      count $N $PI $PJ $TK >> results.log;
    done;

  done;
done;
