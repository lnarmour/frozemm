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

FX="$(cat src/matmultKernel.h | grep '^#define FOOT.*X' | cut -d ' ' -f 3)"
FY="$(cat src/matmultKernel.h | grep '^#define FOOT.*Y' | cut -d ' ' -f 3)"

if [ -n "$SMALL" ]; then
  Is=({1..1} {2..10..2} {20..100..10});
elif [ -n "$MEDIUM" ]; then
  Is=({125..250..25});
else
  Is=({1..1} {250..1000..250});
fi;

for i in ${Is[@]}; 
do
  if [[ $((i*FX)) -gt 10000 ]]; then
    break;
  fi;
  printf "./bin/MM.check $i ";
  err="$(./bin/MM.check $i | grep FAILED)";
  if [ -n "$err" ]; then
    echo "[FAILED]";
  else
    echo "[PASSED]";
  fi
done
