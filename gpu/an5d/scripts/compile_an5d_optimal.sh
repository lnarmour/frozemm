#!/bin/bash

if [[ -z "$SB_TYPE" ]]; then
  SB_TYPE=float;
fi
mkdir -p bin/$SB_TYPE

for l in `cat config/pascal_${SB_TYPE}_configs`; 
do 
  stencil=`echo $l | cut -d ',' -f 1`; 
  bt=`echo $l | cut -d ',' -f 2`; 
  bs=`echo $l | cut -d ',' -f 3`; 
  sl=`echo $l | cut -d ',' -f 4`; 
  regs=`echo $l | cut -d ',' -f 5`; 

  stencil="${stencil}-${bs}-${bt}-${sl}"

  prefix="$(realpath ~)/git/AN5D-Artifact/compiled/${SB_TYPE}/${stencil}";
  host_cu="${prefix}_host.cu";
  kernel_cu="${prefix}_kernel.cu";
  kernel_hu="${prefix}_kernel.hu";

  if [[ ! -f $host_cu || ! -f $kernel_cu || ! -f $kernel_hu ]]; then
    echo error with $stencil;
  fi
  if [[ -n $COPY ]]; then
    cp $host_cu src/${SB_TYPE}/;
    cp $kernel_cu src/${SB_TYPE}/;
    cp $kernel_hu src/${SB_TYPE}/;
  fi
 
  # compile with nvcc and optimal maxrregcount if regs != '-'
  cmd="make SB_TYPE=${SB_TYPE} stencil=${stencil}";
  if [[ ! "$regs" == "-" ]]; then
    cmd="${cmd} regs=-maxrregcount=${regs}";
  fi
  echo $cmd;
  eval $cmd;
done;
