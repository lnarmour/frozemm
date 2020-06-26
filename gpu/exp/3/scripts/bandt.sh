#!/bin/bash

printf "make ";
make clean > /dev/null 2>&1
make > /dev/null 2>&1
if [ "$?" == "0" ]; then
  echo "[PASSED]";
else
  echo "[FAILED]";
  exit 1;
fi;

Ns=(1 5 11 15 50);

for N in ${Ns[@]}; 
do
  printf "./bin/MM.check $N ";
  err="$(./bin/MM.check $N | grep FAILED)";
  if [ -n "$err" ]; then
    echo "[FAILED]";
  else
    echo "[PASSED]";
  fi
done
