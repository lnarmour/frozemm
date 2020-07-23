#!/usr/local/bin/bash

# given a C stencil, and AN5D configuration parameters (bs1, bt, sl)
# generate the cuda source files
if [[ -z "$(which an5d 2>/dev/null)" ]]; then
  echo "evironment error: an5d tool not installed";
  exit 1;
fi

if [[ -z "$1" || -z "$2" || -z "$3" || -z "$4" ]]; then
  echo "usage: $0 STENCIL_C BS1 BT SL";
  exit 1;
fi

bs1=$2
bt=$3
sl=$4

an5d --bs1=$bs1 --bt=$bt --sl=$sl $1

stencil=`echo $1 | sed 's~\(.*\)\..*~\1~'`

if [[ ! -f "${stencil}_kernel.cu" ]]; then
  echo "an5d code generation failed";
  exit 1;
fi

sed -i '' "s~\(#include \"${stencil}\)\(_kernel.hu\"\)~\1-${bs1}-${bt}-${sl}\2~" ${stencil}_host.cu
sed -i '' "s~\(#include \"${stencil}\)\(_kernel.hu\"\)~\1-${bs1}-${bt}-${sl}\2~" ${stencil}_kernel.cu

mv ${stencil}_host.cu ${stencil}-${bs1}-${bt}-${sl}_host.cu
mv ${stencil}_kernel.cu ${stencil}-${bs1}-${bt}-${sl}_kernel.cu
mv ${stencil}_kernel.hu ${stencil}-${bs1}-${bt}-${sl}_kernel.hu

