#!/bin/bash

M=$1
lda=$2
incx=$3
uplo=$4
trans=$5
diag=$6

eval "export USE_LOW_LEVEL=1"
eval "make"
eval "python3 data_gen_htrmv.py $M $lda $incx $uplo $trans $diag"
eval "./htrmv $M $lda $incx $uplo $trans $diag"