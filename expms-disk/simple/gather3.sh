#!/bin/bash

B=(8000000000 4000000000 2000000000 1000000000 500000000 250000000 125000000 62500000 31250000 15625000 7812500 3906250 1953125 976562 488281 244140 122070 61035 30517 15258 7629 3814);

FILE=data/bonito/bonito.log;
rm -rf $FILE;

N=20

for b in ${B[@]}; 
do
  echo "$b | " >> $FILE;

  for r in {1..3};
  do
    t="$(./thru_2MB_3 $N $b)";
    cmd="sed -i '\${s/\$/${t} /}' ${FILE}";
    eval $cmd;
    sleep 1;
  done;

  sed -i '${s/$/| /}' ${FILE}

  for r in {1..3};
  do
    t="$(./thru_1GB_3 $N $b)";
    cmd="sed -i '\${s/\$/${t} /}' ${FILE}";
    eval $cmd;
    sleep 1;
  done;

  sed -i '${s/$/| /}' ${FILE}

  for r in {1..3};
  do
    t="$(./thru_4KB_3 $N $b)";
    cmd="sed -i '\${s/\$/${t} /}' ${FILE}";
    eval $cmd;
    sleep 1;
  done;
  sed -i '${s/$/| /}' ${FILE}


done;
