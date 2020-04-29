#!/bin/bash

B=(2147483648 1073741824 536870912 268435456 134217728 67108864 33554432 16777216 8388608 4194304 2097152 1048576 524288 262144 131072 65536 32768 16384 8192 4096 2048 1024);

FILE=log/4KB.log;
rm -rf $FILE;

for b in ${B[@]}; 
do
  echo "$b | " >> $FILE;
  for hog in {16..20..4};
  do
    for r in {1..5};
    do
      t="$(./thru_4KB $hog 2 $b)";
      cmd="sed -i '\${s/\$/${t}, /}' ${FILE}";
      eval $cmd;
    done;
  done;
done;
