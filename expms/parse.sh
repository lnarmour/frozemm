#!/bin/bash

TK=({5000..500..-500} {475..25..-25})

for tk in ${TK[@]}; 
do 
  t="$(cat results.log | grep -A10 "TK=$tk)" | grep '^.\+\.' | python utils/mean.py)";
  offcore_reqs="$(cat results.log | grep -A20 "TK=$tk)" | grep 'OFFCORE' | sed 's~^[ ]\+\([0-9]\+,.*\)      r.*~\1~' | sed 's~,~~g' )";
  cmd="$(cat results.log | grep "TK=$tk)")"
  N="$(echo $cmd | sed 's~./MM \([0-9]\+\) .*~\1~')"
  PI="$(echo $cmd | sed 's~./MM.*PI=\([0-9]\+\) .*~\1~')"
  PJ="$(echo $cmd | sed 's~./MM.*PJ=\([0-9]\+\) .*~\1~')"

  echo "$N,$PI,$PJ,$tk,$offcore_reqs,$t"
done;
