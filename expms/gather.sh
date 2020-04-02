#!/bin/bash

rm -rf results.log;

N=5000
Ps=({100..2500..100})
TKs=({1000..25..-25})

count $N $N $N $N >> results.log; 

for P in ${Ps[@]}; 
do 
  for TK in ${TKs[@]}; 
  do
    count $N $P $P $TK >> results.log;
  done;
done;
