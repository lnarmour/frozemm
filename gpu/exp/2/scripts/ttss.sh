#!/bin/bash

if [ -z "$1" ]; then
  LOG_FILE='scripts/logs/default.log'
else
  LOG_FILE=$1;
fi

rm -rf $LOG_FILE;

if [ -n "$TENS" ]; then
  N=10000;
  Ps=({500..5000..500});
  TKs=({200..1000..200});
else
  N=10240;
  Ps=({512..5120..512});
  TKs=({256..1024..256});
fi;

if [ -n "$JOULES" ]; then
  for r in {1..3};
  do
    c="./bin/$HOSTNAME/sgemm-ttss.cuda.$CV_NUM $N $N $N $N";
    echo $c >> $LOG_FILE;
    eval $c >> $LOG_FILE;
  done;
  echo "" >> $LOG_FILE;
  
  for P in ${Ps[@]}; 
  do 
    for TK in ${TKs[@]};
    do
      for r in {1..3};
      do
        c="./bin/$HOSTNAME/sgemm-ttss.cuda.$CV_NUM $N $P $P $TK";
        echo $c >> $LOG_FILE;
        eval $c >> $LOG_FILE;
      done;
      echo "" >> $LOG_FILE;
    done;
  done;
fi;

## if [ -n "$NVPROF" ]; then
##   c="lnvprof --profile-from-start off -m dram_read_transactions,dram_write_transactions bin/$HOSTNAME/sgemm-ttss $N $N $N $N";
##   echo $c;
##   eval $c;
##   echo "";
## 
##   for P in ${Ps[@]}; 
##   do 
##     for TK in ${TKs[@]};
##     do
##       c="lnvprof --profile-from-start off -m dram_read_transactions,dram_write_transactions bin/$HOSTNAME/sgemm-ttss $N $P $P $TK";
##       echo $c;
##       eval $c;
##       echo "";
##     done;
##   done;
## fi;
