#!/bin/bash

machine=$(hostname)
log="${machine}.log"

kernels=(GEMM MM);
flags=(novec xhost broadwell haswell ivybridge xavx avx2);
Ns=(2000 5000 7000);

rm -rf ${log};

for kernel in ${kernels[@]};
do
  for N in ${Ns[@]};
  do
    for flag in ${flags[@]};
    do
      printf "${kernel},${N},${machine},${flag}," >> ${log};
      for r in {1..5};
      do
        if [ "${kernel}" == "GEMM" ]; then
          commit="f2fe92d";
        elif [ "${kernel}" == "MM" ]; then
          commit="ec0c401";
        fi
        time="$(./bin/${kernel}.${machine}.${flag}.aligned.${commit} ${N})";
        if [ -z "${time}" ]; then
          printf "illegal instruction,,,,," >> ${log};
          break;
        fi
        printf "${time}," >> ${log};
      done;
      printf "\n" >> ${log};
    done
  done;
done;
