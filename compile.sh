#!/bin/bash

commit="$(git rev-parse --short HEAD)"
flags=(-no-vec -xhost -xavx -xcore-avx2 -march=broadwell -march=haswell -march=ivybridge)

for flag in ${flags[@]};
do
  desc="$(echo $flag | sed 's~-march=~~' | sed 's~-xcore~~' | sed 's~-~~g')"
  make -B OPTS="-O3 ${flag}"
  mv MM "GEMM.$(hostname).${desc}.aligned.${commit}"
  mv ss.optrpt "GEMM.optrpt.$(hostname).${desc}.aligned.${commit}"
done;

