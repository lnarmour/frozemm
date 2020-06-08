#!/bin/bash


METRICS="dram_read_throughput,dram_write_throughput,dram_read_bytes,dram_write_bytes,flop_count_sp";
CMD="nvprof --profile-from-start off -m ${METRICS} ./bin/transpose.fixedflops.nvprof"
F=16384;
MAX=20;
eval "Xs=({1..$MAX})"

for x in ${Xs[@]}; 
do
  N=$((1024*$x));
  S=$(($x**2));  # compModFactor
  echo $CMD $N $F $S;
  eval "$CMD $N $F $S" 2>&1 | grep -A50 'Metric result:';
  echo ""; 
  sleep 5;
done;

CMD="./bin/transpose.fixedflops"

for x in ${Xs[@]};
do
  N=$((1024*$x));
  S=$(($x**2));  # compModFactor
  echo $CMD $N $F $S;
  eval "$CMD $N $F $S";
  echo "";
  sleep 5;
done;
