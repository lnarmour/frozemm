#!/bin/bash

rm -rf results.log;

N=5000
Ps=({100..3000..100})
TKs=({1000..25..-25})

count-energy-pkg $N $N $N $N >> results.log; 
count-energy-ram $N $N $N $N >> results.log; 

for P in ${Ps[@]}; 
do
  for TK in ${TKs[@]}; 
  do
    count-energy-pkg $N $P $P $TK >> results.log;
    count-energy-ram $N $P $P $TK >> results.log;
  done;

done;
