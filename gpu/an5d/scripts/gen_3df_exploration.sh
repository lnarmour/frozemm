#!/usr/local/bin/bash

STENCILS=(star3d1r star3d2r star3d3r star3d4r box3d1r box3d2r box3d3r box3d4r j3d27pt);
BT=({1..5});
BS=(16 32 64);
SL=(128 256 512);
regs=(32 64 96 U);

CNT=0;
for stencil in ${STENCILS[@]}; do
for bs1 in ${BS[@]}; do
for bs2 in ${BS[@]}; do
for bt in ${BT[@]}; do
for sl in ${SL[@]}; do
  #CNT=$((CNT+1));
  #if [[ $CNT == 1 ]]; then exit; fi;
  #"${stencil}-${bs1}x${bs2}-${bt}-${sl},${r}" >> config/pascal_float_configs_ts_explore
  echo "./scripts/gen_cuda.sh artifact/${stencil}.c float ${bt} ${sl} ${bs1} ${bs2}";
  ./scripts/gen_cuda.sh artifact/${stencil}.c float ${bt} ${sl} ${bs1} ${bs2};
done;
done;
done;
done;
done;
