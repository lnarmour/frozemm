#!/usr/local/bin/bash

# given a C stencil, and AN5D configuration parameters (bs1, bt, sl)
# generate the cuda source files
if [[ -z "$(which an5d 2>/dev/null)" ]]; then
  echo "evironment error: an5d tool not installed";
  exit 1;
fi

if [[ -z "$1" || -z "$2" || -z "$3" || -z "$4" || -z "$5" ]]; then
  echo "usage: $0 STENCIL_C SB_TYPE BT SL BS1 [BS2]";
  exit 1;
fi

sb_type=$2
bt=$3
sl=$4
bs1=$5
bs2=$6

dir_name=`realpath $1 | sed 's~\(.*\)/[^/]*~\1~'`;
file_name=`realpath $1 | sed 's~.*/\([^/]*\)$~\1~'`;
stencil=`echo $file_name | sed 's~\(.*\)\..*~\1~'`

if [[ -z $bs2 ]]; then
  bs=$bs1;
else
  bs="${bs1}x${bs2}";
fi

if [[ -z $bs2 ]]; then
  cmd="an5d --bs1=$bs1 --bt=$bt --sl=$sl $file_name";
else
  cmd="an5d --bs1=$bs1 --bs2=$bs2 --bt=$bt --sl=$sl $file_name";
fi
#echo $cmd;
if [[ -z "$VERBOSE" ]]; then cmd="$cmd > /dev/null 2>&1"; fi;
pushd $dir_name > /dev/null;
eval $cmd;
popd > /dev/null;

if [[ "$?" != "0" ]]; then
  echo "${dir_name}/${stencil}-${bs}-${bt}-${sl} an5d code generation failed";
  rm -rf ${dir_name}/${stencil}_*.*u
  exit 1;
fi

sed -i '' "s~\(#include \"${stencil}\)\(_kernel.hu\"\)~\1-${bs}-${bt}-${sl}\2~" ${dir_name}/${stencil}_host.cu
sed -i '' "s~\(#include \"${stencil}\)\(_kernel.hu\"\)~\1-${bs}-${bt}-${sl}\2~" ${dir_name}/${stencil}_kernel.cu

configuration_name="${dir_name}/${stencil}-${bs}-${bt}-${sl}";

mv ${dir_name}/${stencil}_host.cu ${configuration_name}_host.cu
mv ${dir_name}/${stencil}_kernel.cu ${configuration_name}_kernel.cu
mv ${dir_name}/${stencil}_kernel.hu ${configuration_name}_kernel.hu

mkdir -p src-f3d/$sb_type;
#scp ${configuration_name}* maxline:~/git/frozemm/gpu/an5d/src-f3d/float/
mv ${configuration_name}* ./src-f3d/$sb_type/

