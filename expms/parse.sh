#!/bin/bash

TK=({5000..2000..-1000} {1500..500..-500} {400..200..-100} {175..25..-25})

FILE=$1
if [ -z "$1" ]; then
  echo "Usage: ./parse.sh RESULTS_FILE";
  exit 1;
fi

N=5000
P=({100..2500..100})
TK=({1000..25..-25})

tk_header=','
for tk in ${TK[@]};
do
  tk_header="$tk_header,$tk"
done;
echo "${tk_header},${tk_header}"

tmp=.tmp123
tmp_times=.tmp456
rm -rf $tmp $tmp_times

for p in ${P[@]};
do
  miss='';
  time='';
  for tk in ${TK[@]}; 
  do
    cat $FILE | grep -A9 "PI=$p " > $tmp
    cat $tmp | grep -A3 "TK=$tk)" | grep '^.\+\.' > $tmp_times
    if [ -z "$(cat $tmp_times)" ]; then
      continue
    fi
    t="$(cat $tmp_times | python utils/mean.py)";
    std="$(cat $tmp_times | python utils/std.py)";
    L3_miss="$(cat $tmp | grep -A15 "TK=$tk)" | grep 'L3_MISS' | sed 's~^[ ]\+\([0-9]\+,.*\)      ANY.*~\1~' | sed 's~,~~g' )";
    cmd="$(cat $tmp | grep "TK=$tk)")"
    N="$(echo $cmd | sed 's~./MM \([0-9]\+\) .*~\1~')"
    PI="$(echo $cmd | sed 's~./MM.*PI=\([0-9]\+\) .*~\1~')"
    PJ="$(echo $cmd | sed 's~./MM.*PJ=\([0-9]\+\) .*~\1~')"
     
    #echo $cmd
    #echo "$N,$PI,$PJ,$tk,$L3_miss,$t,$std"
    #echo ""
    
    miss="$miss,$L3_miss"
    time="$time,$t"
  done;
  echo "$p,$time,,$miss"
done

rm -rf $tmp $tmp_times
