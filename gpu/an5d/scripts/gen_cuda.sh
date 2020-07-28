#!/usr/local/bin/bash

# given a C stencil, and AN5D configuration parameters (bs1, bt, sl)
# generate the cuda source files
if [[ -z "$(which an5d 2>/dev/null)" ]]; then
  echo "evironment error: an5d tool not installed";
  exit 1;
fi

if [[ -z "$1" || ! -f "$1" || -z "$2" || -z "$3" || -z "$4" ]]; then
  echo "usage: $0 STENCIL_C BS1 BT SL";
  exit 1;
fi

bs1=$2
bt=$3
sl=$4

dir_name=`realpath $1 | sed 's~\(.*\)/[^/]*~\1~'`;
file_name=`realpath $1 | sed 's~.*/\([^/]*\)$~\1~'`;

pushd $dir_name > /dev/null;
an5d --bs1=$bs1 --bt=$bt --sl=$sl $file_name
popd > /dev/null;

stencil=`echo $file_name | sed 's~\(.*\)\..*~\1~'`

if [[ ! -f "${dir_name}/${stencil}_kernel.cu" ]]; then
  echo "an5d code generation failed";
  exit 1;
fi

sed -i '' "s~\(#include \"${stencil}\)\(_kernel.hu\"\)~\1-${bs1}-${bt}-${sl}\2~" ${dir_name}/${stencil}_host.cu
sed -i '' "s~\(#include \"${stencil}\)\(_kernel.hu\"\)~\1-${bs1}-${bt}-${sl}\2~" ${dir_name}/${stencil}_kernel.cu

configuration_name="${dir_name}/${stencil}-${bs1}-${bt}-${sl}";

mv ${dir_name}/${stencil}_host.cu ${configuration_name}_host.cu
mv ${dir_name}/${stencil}_kernel.cu ${configuration_name}_kernel.cu
mv ${dir_name}/${stencil}_kernel.hu ${configuration_name}_kernel.hu

scp ${configuration_name}* maxline:~/git/frozemm/gpu/an5d/src-f3d/float/
