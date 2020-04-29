#!/bin/bash

B=(2147483648 1073741824 536870912 268435456 134217728 67108864 33554432 16777216 8388608 4194304 2097152 1048576 524288 262144 131072 65536 32768 16384 8192 4096 2048 1024);

FILE=log/writes.log;
rm -rf $FILE;

HOG=({16..20..4});
RW='writes'

for b in ${B[@]}; 
do
  echo "$b | " >> $FILE;

  for hog in ${HOG[@]};
  do
    for r in {1..4};
    do
      t="$(./thru_1GB_$RW $hog 2 $b)";
      cmd="sed -i '\${s/\$/${t} /}' ${FILE}";
      eval $cmd;
      sleep 1;
    done;
  done;
  sed -i '${s/$/| /}' ${FILE}
  for hog in ${HOG[@]};
  do
    for r in {1..4};
    do
      t="$(./thru_2MB_$RW $hog 2 $b)";
      cmd="sed -i '\${s/\$/${t} /}' ${FILE}";
      eval $cmd;
      sleep 1;
    done;
  done;
  sed -i '${s/$/| /}' ${FILE}
  for hog in ${HOG[@]};
  do
    for r in {1..4};
    do
      t="$(./thru_4KB_$RW $hog 2 $b)";
      cmd="sed -i '\${s/\$/${t} /}' ${FILE}";
      eval $cmd;
      sleep 1;
    done;
  done;
  sed -i '${s/$/| /}' ${FILE}


done;
