#!/bin/bash

rm -rf results.log;

TK=({5000..500..-500} {475..25..-25})
for tk in ${TK[@]}; 
do 
  count 10000 5000 5000 $tk >> results.log; 
done; 

