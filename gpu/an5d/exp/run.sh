#!/bin/bash

TYPES=(float double);
s_2D=({1024..20480..1024});
s_3D=({128..1024..128});

METRICS="dram_read_bytes,dram_write_bytes"

for SB_TYPE in ${TYPES[@]};
do
  for line in `cat config/pascal_${SB_TYPE}_configs`;
  do
    stencil=`echo $line | cut -d ',' -f 1`;
    R=`echo $line | cut -d ',' -f 2`;
    MAXRREGCOUNT=`if [[ "$R" != "U" ]]; then echo "regs=-maxrregcount=$R"; else echo ""; fi;`;
    make_cmd="make stencil=$stencil SB_TYPE=$SB_TYPE REGCOUNT=$R $MAXRREGCOUNT";
    binary_name="./bin/${SB_TYPE}/$(echo $line | sed 's~,~.r~')";

    # make the binary if it doesn't exist
    if [[ ! -f $binary_name || -n "$FORCE_MAKE" ]]; then
      rm -rf $binary_name;
      echo "$make_cmd";
      eval "$make_cmd > /dev/null 2>&1";
    fi
    if [[ "$?" != 0 || ! -f $binary_name ]]; then continue; fi; 

    if [[ "$stencil" == *"2d"* ]]; then 
      S=({1024..20480..1024});
    else
      S=({128..1024..128});
    fi;

    for s in ${S[@]};
    do
      cmd="./scripts/single.sh $binary_name $s";
      if [[ -n ${COLLECT} ]]; then 
        eval $cmd; 
      fi;
    done
  done;
done;
