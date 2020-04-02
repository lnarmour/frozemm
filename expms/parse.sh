#!/bin/bash

TK=({5000..2000..-1000} {1500..500..-500} {400..200..-100} {175..25..-25})

FILE=$1
if [ -z "$1" ]; then
  echo "Usage: ./parse.sh RESULTS_FILE";
  exit 1;
fi

PI=(10000 7500 5000 2500)

for pi in ${PI[@]};
do 
  if [[ "$pi" == "10000" ]]; then
    TK=(10000)
  elif [[ "$pi" == "7500" ]]; then
    TK=({7500..2500..-1000} {2000..500..-500} {400..200..-100} {175..25..-25})
  elif [[ "$pi" == "5000" ]]; then
    TK=({5000..2000..-1000} {1500..500..-500} {400..200..-100} {175..25..-25})
  elif [[ "$pi" == "2500" ]]; then
    TK=({2500..500..-500} {400..200..-100} {175..25..-25})
  fi

  for tk in ${TK[@]}; 
  do 
    if [ -z "$(cat $FILE | grep -A12 "PI=$pi" | grep -A10 "TK=$tk)" | grep '^.\+\.')" ]; then
      continue
    fi
    t="$(cat $FILE | grep -A12 "PI=$pi" | grep -A5 "TK=$tk)" | grep '^.\+\.' | python utils/mean.py)";
    std="$(cat $FILE | grep -A12 "PI=$pi" | grep -A5 "TK=$tk)" | grep '^.\+\.' | python utils/std.py)";
    L3_miss="$(cat $FILE | grep -A12 "PI=$pi" | grep -A16 "TK=$tk)" | grep 'L3_MISS' | sed 's~^[ ]\+\([0-9]\+,.*\)      ANY.*~\1~' | sed 's~,~~g' )";
    cmd="$(cat $FILE | grep -A12 "PI=$pi" | grep "TK=$tk)")"
    N="$(echo $cmd | sed 's~./MM \([0-9]\+\) .*~\1~')"
    PI="$(echo $cmd | sed 's~./MM.*PI=\([0-9]\+\) .*~\1~')"
    PJ="$(echo $cmd | sed 's~./MM.*PJ=\([0-9]\+\) .*~\1~')"
  
    echo "$N,$PI,$PJ,$tk,$L3_miss,$t,$std"
  done;
done
