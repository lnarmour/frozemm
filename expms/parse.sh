#!/bin/bash

TK=({5000..2000..-1000} {1500..500..-500} {400..200..-100} {175..25..-25})

FILE=$1
if [ -z "$1" ]; then
  echo "Usage: ./parse.sh RESULTS_FILE";
  exit 1;
fi

for tk in ${TK[@]}; 
do 
  if [ -z "$(cat $FILE | grep -A10 "TK=$tk)" | grep '^.\+\.')" ]; then
    continue
  fi
  t="$(cat $FILE | grep -A10 "TK=$tk)" | grep '^.\+\.' | python utils/mean.py)";
  std="$(cat $FILE | grep -A10 "TK=$tk)" | grep '^.\+\.' | python utils/std.py)";
  offcore_reqs="$(cat $FILE | grep -A20 "TK=$tk)" | grep 'OFFCORE' | sed 's~^[ ]\+\([0-9]\+,.*\)      r.*~\1~' | sed 's~,~~g' )";
  L3_miss="$(cat $FILE | grep -A21 "TK=$tk)" | grep 'L3_MISS' | sed 's~^[ ]\+\([0-9]\+,.*\)      r.*~\1~' | sed 's~,~~g' )";
  cmd="$(cat $FILE | grep "TK=$tk)")"
  N="$(echo $cmd | sed 's~./MM \([0-9]\+\) .*~\1~')"
  PI="$(echo $cmd | sed 's~./MM.*PI=\([0-9]\+\) .*~\1~')"
  PJ="$(echo $cmd | sed 's~./MM.*PJ=\([0-9]\+\) .*~\1~')"

  echo "$N,$PI,$PJ,$tk,$offcore_reqs,$L3_miss,$t,$std"
done;
