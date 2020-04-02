#!/bin/bash

TK=({5000..2000..-1000} {1500..500..-500} {400..200..-100} {175..25..-25})

FILE=$1
if [ -z "$1" ]; then
  echo "Usage: ./parse.sh RESULTS_FILE";
  exit 1;
fi

N=5000
P=({1000..2500..250})
TK=({1000..100..-100})

tk_header=','
for tk in ${TK[@]};
do
  tk_header="$tk_header,$tk"
done;
echo "${tk_header}${tk_header}"

for p in ${P[@]};
do
  miss='';
  time='';
  for tk in ${TK[@]}; 
  do
    if [ -z "$(cat $FILE | grep -A11 "PI=$p" | grep -A5 "TK=$tk)" | grep '^.\+\.')" ]; then
      continue
    fi

    t="$(cat $FILE | grep -A11 "PI=$p" | grep -A5 "TK=$tk)" | grep '^.\+\.' | python utils/mean.py)";
    std="$(cat $FILE | grep -A11 "PI=$p" | grep -A5 "TK=$tk)" | grep '^.\+\.' | python utils/std.py)";
    L3_miss="$(cat $FILE | grep -A11 "PI=$p" | grep -A18 "TK=$tk)" | grep 'L3_MISS' | sed 's~^[ ]\+\([0-9]\+,.*\)      ANY.*~\1~' | sed 's~,~~g' )";
    cmd="$(cat $FILE | grep -A11 "PI=$p" | grep "TK=$tk)")"
    N="$(echo $cmd | sed 's~./MM \([0-9]\+\) .*~\1~')"
    PI="$(echo $cmd | sed 's~./MM.*PI=\([0-9]\+\) .*~\1~')"
    PJ="$(echo $cmd | sed 's~./MM.*PJ=\([0-9]\+\) .*~\1~')"
     
    #echo $cmd
    #echo "$N,$PI,$PJ,$tk,$L3_miss,$t,$std"
    #echo ""
    
    miss="$miss,$L3_miss"
    time="$time,$t"
  done;
  echo "$p,$time,$miss"
done
