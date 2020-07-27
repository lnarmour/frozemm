#!/bin/bash

EXP_DIR=exp;
METRICS="dram_read_bytes,dram_write_bytes";
j_gb="0.205";


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

function get_gflops {
  bin=$1;
  s=$2;
  SB_TYPE=$3;
  gflops=`run $bin $s $SB_TYPE | cut -d ',' -f 4`;
  exec_time=`run $bin $s $SB_TYPE | cut -d ',' -f 5`;
  echo "$exec_time ms, $gflops GFLOPS/s";
}

function get_dram_energy_percentage {
  total_joules=$1;
  gbs_xfered=$2;
  if [[ -n "$gbs_xfered" && -n "$total_joules" ]]; then
    calc 100*$gbs_xfered*$j_gb/$total_joules;
  fi
}

function run {
  bin=$1;
  s=$2;
  sb_type=$3;
  NVPROF=$4;

  cmd="$bin -n 1 -s $s -t 1000";
  nvprof_cmd="nvprof --profile-from-start off -m $METRICS $cmd";

  stencil=`echo $cmd | sed "s~./bin.*/${sb_type}/\(.*\) -n.*~\1~"`;
  mkdir -p "./${EXP_DIR}/${sb_type}/nvprof";
  if [[ -z $NVPROF ]]; then
    F="./${EXP_DIR}/${sb_type}/${stencil}.s${s}.log";
    eval "$cmd > $F";
  else
    F="./${EXP_DIR}/${sb_type}/nvprof/${stencil}.s${s}.log";
    eval "$nvprof_cmd > $F 2>&1";
  fi



  if [[ -z $NVPROF ]]; then
    line="$(echo $F | cut -d '/' -f 4) $(cat $F | grep Average)";
    line="$(echo $line | sed 's~[	][	]*~~g' | sed 's~,[ ][ ]*~,~g' | sed 's~ Average: ~,~')"
    stencil=`echo $line | cut -d ',' -f 1 | cut -d ' ' -f 1`;
    gflops=`echo $line | cut -d ',' -f 2 | cut -d ' ' -f 1`;
    ms=`echo $line | cut -d ',' -f 3 | cut -d ' ' -f 1`;
    joules=`echo $line | cut -d ',' -f 4 | cut -d ' ' -f 1`;
    TOTAL_JOULES="$(echo $joules)";
    echo "------------- stencil: ${stencil}";
    echo "            precision: ${sb_type}";
    echo "              compute: $gflops GFLOPS/s";
    echo "                 time: $ms ms";
    echo "               energy: $joules Joules";
  else
    TOTAL_GBs="$(parse_nvprof_log $F)";
    echo "         bytes xfered: $TOTAL_GBs GB";
    echo "    % energy on xfers: $(get_dram_energy_percentage $TOTAL_JOULES $TOTAL_GBs)%";
  fi
}

if [[ -z "$1" || -z "$2" ]]; then
  echo "usage: $0 BINARY N";
  exit 1;
fi

bin=$1
s=$2
SB_TYPE="$(if [[ "$bin" == *"float"* ]]; then echo "float"; else echo "double"; fi)"

#echo "$(get_dram_energy_percentage $bin $s $SB_TYPE)  -->  $(get_gflops $bin $s $SB_TYPE)";

TOTAL_JOULES=0;

run $bin $s $SB_TYPE;
run $bin $s $SB_TYPE "nvprof";
