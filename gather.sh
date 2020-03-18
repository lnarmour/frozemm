#!/bin/bash

machine=$(hostname)
log="${machine}.log"

flags=(novec xhost broadwell haswell ivybridge xavx avx2)
Ns=(2000 5000 7000)

rm -rf ${log}

for N in ${Ns[@]};
do
  for flag in ${flags[@]};
  do
    printf "${N},${machine},${flag}," >> ${log};
    for r in {1..5};
    do
      time="$(./bin/GEMM.${machine}.${flag}.aligned.f2fe92d ${N})";
      printf "${time}," >> ${log};
    done;
    printf "\n" >> ${log};
  done
done;
