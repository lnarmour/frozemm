#!/bin/bash

float=`ls -l exp/float/*.log 2>/dev/null | wc -l`;
float_nv=`ls -l exp/float/nvprof/*.log 2>/dev/null | wc -l`;
double=`ls -l exp/double/*.log 2>/dev/null | wc -l`;
double_nv=`ls -l exp/double/nvprof/*.log 2>/dev/null | wc -l`;

echo "float      $float";
echo "float_nv   $float";
echo "double     $double";
echo "double_nv  $double";
