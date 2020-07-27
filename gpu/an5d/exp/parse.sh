#!/bin/bash

j_gb='0.205';

TYPES=(float double);
DIM=("2d" "3d");
s_2d=({1024..20480..1024});
s_3d=({128..1024..128});

METRICS="dram_read_bytes,dram_write_bytes"

calc() { awk "BEGIN { print "$*" }"; }

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

function get_gflops {
  echo `run $1 $2 $3 | cut -d ',' -f 4`;
}

function get_dram_energy_percentage {
  bin=$1;
  s=$2;
  SB_TYPE=$3;

  total_joules=`run $bin $s $SB_TYPE | cut -d ',' -f 6`;
  gbs_xfered=`run $bin $s $SB_TYPE "nvprof"`;
  if [[ -n "$gbs_xfered" && -n "$total_joules" ]]; then
    calc 100*$gbs_xfered*$j_gb/$total_joules;
  fi
}

function check_log {
  F=$1

  if [[ ! -f $F ]]; then echo no-exist; return; fi;
  if [[ "$(cat $F | wc -l)" == 0 ]]; then echo empty; return; fi
  if [[ -n "$(cat $F | grep -i error)" ]]; then echo has-err; return; fi
  
}

function validate {
  bin=$1;
  s=$2;
  SB_TYPE=$3;
  
  stencil=`echo $bin | sed "s~./bin/${SB_TYPE}/\(.*\)~\1~"`;

  F1="./exp/${SB_TYPE}/${stencil}.s${s}.log";
  F2="./exp/${SB_TYPE}/nvprof/${stencil}.s${s}.log";
  err="$(check_log $F1)"; if [[ -n "$err" ]]; then printf "1-$err,"; fi;
  err="$(check_log $F2)"; if [[ -n "$err" ]]; then printf "2-$err,"; fi;

}


for SB_TYPE in ${TYPES[@]};
do
  for dim in ${DIM[@]};
  do
    S=();
    eval "for s in \${s_${dim}[@]}; do S+=(\$s); done;"
    for bin in ./bin/${SB_TYPE}/*${dim}*;
    do
      if [[ ! -f "$bin" ]]; then continue; fi;
      printf "${SB_TYPE},$(echo $bin | cut -d '/' -f 4 | sed 's~\(.*\)~\1~'),";
      for s in ${S[@]};
      do
        if [[ -n "$(validate $bin $s $SB_TYPE)" ]]; then printf ","; continue; fi;
        printf "$(get_dram_energy_percentage $bin $s $SB_TYPE),"       
        #printf "$(get_gflops $bin $s $SB_TYPE),"
      done;
      printf "\n";
    done;
  done
done;
