#!/bin/bash

SRC=fma.MM2.c
N=2000

eval "for v in {$S..$E}; do icc \${SRC} -o FMA.v\${v} -O3 -std=c99 -I/usr/include/malloc/ -xcore-avx2 -Dversion\${v} > /dev/null 2>&1; printf \"version\${v}\"; for run in {0..1}; do printf \",\$(./FMA.v\${v} \${N})\"; done; printf \"\n\"; rm -rf FMA.v\${v}; done;"
