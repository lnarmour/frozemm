#!/bin/bash

rm -rf results.log;

count 10000 10000 10000 10000 >> results.log; 

TK=({7500..2500..-1000} {2000..500..-500} {400..200..-100} {175..25..-25})
for tk in ${TK[@]}; 
do 
  count 10000 7500 7500 $tk >> results.log; 
done; 

TK=({5000..2000..-1000} {1500..500..-500} {400..200..-100} {175..25..-25})
for tk in ${TK[@]}; 
do 
  count 10000 5000 5000 $tk >> results.log; 
done; 

TK=({2500..500..-500} {400..200..-100} {175..25..-25})
for tk in ${TK[@]}; 
do 
  count 10000 2500 2500 $tk >> results.log; 
done; 

