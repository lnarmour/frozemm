#!/bin/bash

if [[ -n "$1" ]]; then
  EVAL=$1;
fi

function eval_cmd {
  cmd=$1;
  do_EVAL=$2;
  CNT=$((CNT+1));
  if [[ -n $MAX ]]; then out_of="/$MAX"; DATE=" $(date "+%Y-%m-%d:%H:%M:%S")"; fi;
  echo "${CNT}${out_of}${DATE} '$cmd'";
  if [[ -z "$VERBOSE" ]]; then cmd="$cmd > /dev/null 2>&1"; fi;
  if [[ -n "$do_EVAL" ]]; then 
    eval $cmd; 
    if [[ "$?" != "0" ]]; then echo "^nvcc error"; fi;
  else
    echo "";
  fi;
}

function gen {
  for line in `cat config/ts_explore_configs`;
  do stencil=`echo $line | cut -d ',' -f 1`;
  
    bt=`echo $line | cut -d ',' -f 2`; 
    sl=`echo $line | cut -d ',' -f 3`; 
    regs=`echo $line | cut -d ',' -f 4`;
    sb_type=`echo $line | cut -d ',' -f 5`;
    bs1=`echo $line | cut -d ',' -f 6`;
    bs2=`echo $line | cut -d ',' -f 7`;
    if [[ -n "$bs2" ]]; then 
      bs="${bs1}x${bs2}"; 
    else 
      bs=$bs1; 
    fi; 
    
    for R in {32..96..32};
    do
      eval_cmd "make stencil=${stencil}-${bs}-${bt}-${sl} SB_TYPE=${sb_type} regs='-maxrregcount=${R}' REGCOUNT=${R}" $EVAL; 
    done;
    eval_cmd "make stencil=${stencil}-${bs}-${bt}-${sl} SB_TYPE=${sb_type} REGCOUNT=U" $EVAL; 
  done;
}

CNT=0;
MAX=`gen | wc -l | sed 's~[ ][ ]*\(.*\)~\1~'`;

gen $EVAL;
