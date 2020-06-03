#!/bin/bash

# the below is written based on the state of src/transpose.cu at revision:
# 09fd9cda08d71bf1636bb7f85533f1256af21c8f

LOG=scripts/logs/transpose.N.FMAS.log;
mkdir -p scripts/logs;
rm -rf $LOG;

for N in {1024..10240..1024};
do
  for F in {4..100..4};
  do
    echo ./bin/transpose $N $F >> $LOG;
    ./bin/transpose $N $F >> $LOG;
    echo "" >> $LOG;
  done;

  for F in {250..1000..250};
  do
    echo ./bin/transpose $N $F >> $LOG;
    ./bin/transpose $N $F >> $LOG;
    echo "" >> $LOG;
  done;
done;
