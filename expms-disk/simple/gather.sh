#!/bin/bash

B=(8589934592 4294967296 2147483648 1073741824 536870912 268435456 134217728 67108864 33554432 16777216 8388608 4194304 2097152 1048576 524288 262144 131072 65536 32768 16384 8192 4096);

FILE=data/bullhead/bullhead.log;
rm -rf $FILE;

N=20

for b in ${B[@]}; 
do
  echo "$b | " >> $FILE;

  for r in {1..3};
  do
    t="$(./thru_2MB_2 $N $b)";
    cmd="sed -i '\${s/\$/${t} /}' ${FILE}";
    eval $cmd;
    sleep 1;
  done;

  sed -i '${s/$/| /}' ${FILE}

  for r in {1..3};
  do
    t="$(./thru_1GB_2 $N $b)";
    cmd="sed -i '\${s/\$/${t} /}' ${FILE}";
    eval $cmd;
    sleep 1;
  done;

  sed -i '${s/$/| /}' ${FILE}

  for r in {1..3};
  do
    t="$(./thru_4KB_2 $N $b)";
    cmd="sed -i '\${s/\$/${t} /}' ${FILE}";
    eval $cmd;
    sleep 1;
  done;
  sed -i '${s/$/| /}' ${FILE}


done;
