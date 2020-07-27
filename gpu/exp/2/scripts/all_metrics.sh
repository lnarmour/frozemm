#!/bin/bash

if [ -z "$1" ]; then
  echo "usage: $0 grep_string";
  exit 1;
fi

str="";
for p in $(cat scripts/nvprof.metrics | cut -d '|' -f 1 | grep $1); 
do 
  str+=",$p"; 
done; 
if [ -n "$str" ]; then
  echo $str | sed 's~,\(.*\)~\1~g';
fi
