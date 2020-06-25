#!/bin/bash

N=10000;

if [ -n "$JOULES" ]; then
  for r in {1..3};
  do
    c="./bin/sgemm-ttss $N $N $N $N";
    echo $c;
    eval $c;
  done;
  echo "";
  
  for P in {500..5000..500}; 
  do 
    for TK in {100..1000..100};
    do
      for r in {1..3};
      do
        c="./bin/sgemm-ttss $N $P $P $TK";
        echo $c;
        eval $c;
      done;
      echo "";
    done;
  done;
fi;

if [ -n "$NVPROF" ]; then
  c="lnvprof --profile-from-start off -m dram_read_transactions,dram_write_transactions bin/sgemm-ttss $N $N $N $N";
  echo $c;
  eval $c;
  echo "";

  for P in {500..5000..500}; 
  do 
    for TK in {100..1000..100};
    do
      c="lnvprof --profile-from-start off -m dram_read_transactions,dram_write_transactions bin/sgemm-ttss $N $P $P $TK";
      echo $c;
      eval $c;
      echo "";
    done;
  done;
fi;
