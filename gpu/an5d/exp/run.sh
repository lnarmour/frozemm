#!/bin/bash

TYPES=(float double);
s_2D=({1024..20480..1024});
s_3D=({128..1024..128});

METRICS="dram_read_bytes,dram_write_bytes"

function run {
  bin=$1;
  s=$2;
  sb_type=$3;
  cmd="$bin -n 1 -s $s -t 1000";
  nvprof_cmd="nvprof --profile-from-start off -m $METRICS $cmd";

  stencil=`echo $cmd | sed "s~./bin/${sb_type}/\(.*\) -n.*~\1~"`;
  
  C1="$cmd > ./exp/${sb_type}/${stencil}.s${s}.log";
  C2="$nvprof_cmd > ./exp/${sb_type}/nvprof/${stencil}.s${s}.log 2>&1";

  echo $C1;
  eval $C1;
  echo $C2;
  eval $C2;
}


for SB_TYPE in ${TYPES[@]};
do
  mkdir -p ./exp/${SB_TYPE}/nvprof;
  for bin in ./bin/${SB_TYPE}/*2d*;
  do
    for s in ${s_2D[@]};
    do
      run $bin $s $SB_TYPE;
    done;
  done;

  for bin in ./bin/${SB_TYPE}/*3d*;
  do
    for s in ${s_3D[@]};
    do
      run $bin $s $SB_TYPE;
    done;
  done;

done;
