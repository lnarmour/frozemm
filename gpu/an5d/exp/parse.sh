#!/bin/bash

TYPES=(float double);
s_2D=({1024..20480..1024});
s_3D=({128..1024..128});

METRICS="dram_read_bytes,dram_write_bytes"

calc(){ awk "BEGIN { print "$*" }"; }

function parse_nvprof_log {
  log_file=$1;

  kernel_calls=`cat $log_file | grep dram_.*_bytes | sed 's~[ ][ ]*~ ~g' | cut -d ' ' -f 2`
  bytes=`cat $log_file | grep dram_.*_bytes | sed 's~[ ][ ]*~ ~g' | cut -d ' ' -f 14`
  paste -d '|' <(printf "%s\n" $kernel_calls) <(printf "%s\n" $bytes) | sed 's~\(.*\)|\(.*\)~total_bytes=`calc $total_bytes+\1*\2`~' > /tmp/.tmp-an5d-parse.txt

  total_bytes=0
  while read l; 
  do
    eval $l;
  done < /tmp/.tmp-an5d-parse.txt
  giga='1000000000';
  total_GBs=`calc $total_bytes/$giga`;
  echo $total_GBs;
  rm -rf /tmp/.tmp-an5d-parse.txt;
}


function run {
  bin=$1;
  s=$2;
  sb_type=$3;
  NVPROF=$4;

  cmd="$bin -n 1 -s $s -t 1000";
  nvprof_cmd="nvprof --profile-from-start off -m $METRICS $cmd";

  stencil=`echo $cmd | sed "s~./bin/${sb_type}/\(.*\) -n.*~\1~"`;
  
  if [[ -z $NVPROF ]]; then
    F="./exp/${sb_type}/${stencil}.s${s}.log";
  else
    F="./exp/${sb_type}/nvprof/${stencil}.s${s}.log";
  fi

  if [[ ! -f $F ]]; then
    return;
  fi

  if [[ -z $NVPROF ]]; then
    line="$(echo $F | sed 's~.*/\([^-]*\)-.*~\1~') $(cat $F | grep Average)";
    line="$(echo $line | sed 's~[	][	]*~~g' | sed 's~,[ ][ ]*~,~g' | sed 's~ Average: ~,~')"  
    stencil=`echo $line | cut -d ',' -f 1 | cut -d ' ' -f 1`;
    gflops=`echo $line | cut -d ',' -f 2 | cut -d ' ' -f 1`;
    ms=`echo $line | cut -d ',' -f 3 | cut -d ' ' -f 1`;
    joules=`echo $line | cut -d ',' -f 4 | cut -d ' ' -f 1`;
    echo "${sb_type},${stencil},$s,$gflops,$ms,$joules";
  else
    parse_nvprof_log $F
  fi
}

function parse {
  rm -rf /tmp/.tmp-an5d-joules.txt /tmp/.tmp-an5d-nvprof.txt
  touch /tmp/.tmp-an5d-joules.txt /tmp/.tmp-an5d-nvprof.txt
  run $1 $2 $3 > /tmp/.tmp-an5d-joules.txt
  run $1 $2 $3 "nvprof" > /tmp/.tmp-an5d-nvprof.txt
  paste -d ',' /tmp/.tmp-an5d-joules.txt /tmp/.tmp-an5d-nvprof.txt
}


for SB_TYPE in ${TYPES[@]};
do
  for bin in ./bin/${SB_TYPE}/*2d*;
  do
    for s in ${s_2D[@]};
    do
      parse $bin $s $SB_TYPE;
    done;
  done;

  for bin in ./bin/${SB_TYPE}/*3d*;
  do
    for s in ${s_3D[@]};
    do
      parse $bin $s $SB_TYPE;
    done;
  done;

done;
