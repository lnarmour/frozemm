#!/bin/bash

# ./bin/transpose.nomem 10240 1000
# ./bin/transpose.fixedmem 10240 0
# ./bin/transpose.fixedmem 10240 1000

METRICS="dram_read_bytes,dram_write_bytes,flop_count_sp";
CMD0="nvprof --profile-from-start off -m ${METRICS} ./bin/transpose.nomem.nvprof"
CMD1="nvprof --profile-from-start off -m ${METRICS} ./bin/transpose.fixedmem.nvprof"
MAX=20;
eval "Xs=({1..$MAX})"

# F=10;
# for x in ${Xs[@]}; 
# do
#   N=$((1024*$x));
#   echo $CMD0 $N $F;
#   eval "$CMD0 $N $F" 2>&1 | grep -A50 'Metric result:';
#   echo $CMD1 $N 0;
#   eval "$CMD1 $N 0 " 2>&1 | grep -A50 'Metric result:';
#   echo $CMD1 $N $F;
#   eval "$CMD1 $N $F" 2>&1 | grep -A50 'Metric result:';
#   echo ""; 
# done;

CMD0="./bin/transpose.nomem"
CMD1="./bin/transpose.fixedflops"

F=65536;
for x in ${Xs[@]};
do
  N=$((1024*$x));
  S=$(($x**2))
  #echo $CMD0 $N $F $S;
  #eval "$CMD0 $N $F $S";
  #echo $CMD1 $N 0;
  #eval "$CMD1 $N 0";
  echo $CMD1 $N $F $S;
  eval "$CMD1 $N $F $S";
  echo "";
done;
