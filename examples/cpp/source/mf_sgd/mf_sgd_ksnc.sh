#!/bin/bash

mbind=1
# mbind=0
rate=0.0001
# rate=0.005
lambda=1
# lambda=0.003
nItr=10
# thds=256
thds=64
grain=10000
dim=128
avx=0

path=/home/langshichen/Lib/__release_lnx/daal/examples/cpp/_results/intel_intel64_parallel_a
export I_MPI_DEBUG=5
source /opt/intel/impi/2017.0.098/intel64/bin/mpivars.sh

# for ((nItr = 256; nItr>1; nItr /= 2));  

# do
		
  mpirun -host KnlRed -np 4 $path/mf_sgd_ksnc.exe $rate $lambda $nItr $thds $grain $dim $avx >> $path/mf_sgd_ksnc-$mbind-$rate-$lambda-$nItr-$thds-$grain-$dim-$avx.res

# done

