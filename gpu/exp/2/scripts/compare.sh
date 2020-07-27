#!/bin/bash

if [[ -z "$1" || -z "$2" ]]; then
  echo "usage: $0 metric1[,metric2[,metric3...] N [PI PJ TK]";
  exit 1;
fi

M=$1;

args="$@";
args="$(echo $args | sed "s~$1 \(.*\)~\1~")"
eval set -- "$args";

set_cuda_version cuda9_1;
make clean > /dev/null 2>&1;
make > /dev/null 2>&1;
c="lnvprof --profile-from-start off -m $M bin/$HOSTNAME/sgemm-ttss.cuda.9.1 $@";
#echo $c;
eval $c;

echo "";

set_cuda_version cuda;
make clean > /dev/null 2>&1;
make > /dev/null 2>&1;
c="lnvprof --profile-from-start off -m $M bin/$HOSTNAME/sgemm-ttss.cuda.10.1 $@";
#echo $c;
eval $c;
