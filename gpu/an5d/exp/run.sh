#!/bin/bash

# example usage:
# 
# to run the configs in a config file using directories src2 and bin2
# COLLECT=1 ./exp/run.sh 2 config/pascal_double_configs_ts_explore 
# 
# to run normally using both config files config/pascal_*_configs
# COLLECT=1 ./exp/run.sh
#
# run without COLLECT to compile only
# use FORCE_MAKE to recompile
#


TYPES=(float double);
s_2D=({1024..20480..1024});
s_3D=({128..1024..128});

METRICS="dram_read_bytes,dram_write_bytes"

if [[ -n "$1" ]]; then
  # used to control which src and bin directories to use
  ROUND=$1;
fi

function process_line {
  SB_TYPE=$1;
  line=$2;
  stencil=`echo $line | cut -d ',' -f 1`;
  R=`echo $line | cut -d ',' -f 2`;
  MAXRREGCOUNT=`if [[ "$R" != "U" ]]; then echo "regs=-maxrregcount=$R"; else echo ""; fi;`;
  make_cmd="make stencil=$stencil SB_TYPE=$SB_TYPE REGCOUNT=$R ROUND=$ROUND $MAXRREGCOUNT";
  binary_name="./bin${ROUND}/${SB_TYPE}/$(echo $line | sed 's~,~.r~')";

  # make the binary if it doesn't exist
  if [[ ! -f $binary_name || -n "$FORCE_MAKE" ]]; then
    rm -rf $binary_name;
    echo "$make_cmd";
    eval "$make_cmd > /dev/null 2>&1";
  fi
  if [[ "$?" != 0 || ! -f $binary_name ]]; then return; fi; 

  S=(); 
  if [[ "$stencil" == *"2d"* ]]; then 
    for s in ${s_2D[@]}; do S+=($s); done;
  else
    for s in ${s_3D[@]}; do S+=($s); done;
  fi;

  for s in ${S[@]};
  do
    cmd="./scripts/single.sh $binary_name $s";
    if [[ -n ${COLLECT} ]]; then 
      eval $cmd; 
    fi;
  done
}

if [[ -z "$2" ]]; then
  for SB_TYPE in ${TYPES[@]};
  do
    for line in `cat config/pascal_${SB_TYPE}_configs | grep -v "^#"`;
    do
      process_line $SB_TYPE $line;
    done;
  done;
else
  s_2D=({16384..16384});
  s_3D=({512..512});
  for line in `cat $2 | grep -v "^#"`;
  do
    SB_TYPE=`if [[ "float" == *"$2"* ]]; then echo "float"; else echo "double"; fi;`
    process_line $SB_TYPE $line;
  done;
fi
