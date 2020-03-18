#!/bin/bash

commit="$(git rev-parse --short HEAD)"
flags=(-no-vec -xhost -xavx -xcore-avx2 -march=broadwell -march=haswell -march=ivybridge)

for flag in ${flags[@]};
do
  desc="$(echo $flag | sed 's~-march=~~' | sed 's~-xcore~~' | sed 's~-~~g')"
  make -B OPTS="-O3 ${flag}"
  mv MM "MM.$(hostname).${desc}.unaligned.${commit}"
  mv ss.optrpt "MM.optrpt.$(hostname).${desc}.unaligned.${commit}"
done;

