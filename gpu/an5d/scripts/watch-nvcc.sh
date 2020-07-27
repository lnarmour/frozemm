#!/bin/bash


float_cnt=`ls -l bin/float/ | grep '\.r' | wc -l`;
double_cnt=`ls -l bin/double/ | grep '\.r' | wc -l`;

curr=`ps aux | grep lnarmour | grep 'make stencil' | grep SB_TYPE | sed 's~.*\(make stencil.*\)~\1~'`;

echo "current process: ${curr}";
echo "          float: ${float_cnt}";
echo "         double: ${double_cnt}";

