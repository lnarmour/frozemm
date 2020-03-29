#!/bin/bash

rm -rf results.log;

TK=({5000..2000..-1000} {1500..500..-500} {400..200..-100} {175..25..-25})
for tk in ${TK[@]}; 
do 
  count 10000 5000 5000 $tk >> results.log; 
done; 

