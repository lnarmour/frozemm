#!/bin/bash

versions=(cuda9_1 cuda);
for v in ${versions[@]};
do
  set_cuda_version $v;
  make clean > /dev/null 2>&1;
  make > /dev/null 2>&1;
  export JOULES=1
  TENS=1 ./scripts/ttss.sh "scripts/logs/ttss.joules.$HOSTNAME.cuda.$CV_NUM.m1000s.log"
  ./scripts/ttss.sh "scripts/logs/ttss.joules.$HOSTNAME.cuda.$CV_NUM.m1024s.log"
done;
