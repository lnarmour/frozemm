#!/bin/bash

DIR=exp2

float=`ls -l $DIR/float/*.log 2>/dev/null | wc -l`;
float_nv=`ls -l $DIR/float/nvprof/*.log 2>/dev/null | wc -l`;
double=`ls -l $DIR/double/*.log 2>/dev/null | wc -l`;
double_nv=`ls -l $DIR/double/nvprof/*.log 2>/dev/null | wc -l`;

echo "float      $float";
echo "float_nv   $float";
echo "double     $double";
echo "double_nv  $double";
