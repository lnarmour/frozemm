#!/usr/local/bin/bash

if [[ -n "$1" ]]; then
  EVAL=$1;
fi

BS2d=(128 256 512);
BS3d=(16 32 64);
BT=(1 2 4 8);
SL=(128 256 512);
regs=(32 64 96);

function eval_cmd {
  cmd=$1;
  EVAL=$2;
  CNT=$((CNT+1));
  if [[ -n $MAX ]]; then out_of="/$MAX"; fi;
  echo "[${CNT}${out_of}] $cmd"; 
  if [[ -n "$EVAL" ]]; then 
    eval $cmd; 
  fi;
}

function gen {
  EVAL=$1;
  for bt in ${BT[@]}; do
  for sl in ${SL[@]}; do
  for bs in ${BS2d[@]}; do
    # 2D
    eval_cmd "./gen_cuda.sh box2d2r.c double $bt $sl $bs" $EVAL;
  done;
  for bs1 in ${BS3d[@]}; do
    for bs2 in ${BS3d[@]}; do
      # 3D
      eval_cmd "./gen_cuda.sh star3d4r.c float $bt $sl $bs1 $bs2" $EVAL;
    done;
  done;
  done;
  done;
}

CNT=0;
MAX=`gen | wc -l | sed 's~[ ][ ]*\(.*\)~\1~'`;

cp ~/git/AN5D-Artifact/*.c .;

gen $EVAL;
