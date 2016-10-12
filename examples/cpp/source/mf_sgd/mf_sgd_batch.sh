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

# for ((nItr = 256; nItr>1; nItr /= 2));  

# do
		
    numactl --membind $mbind $path/mf_sgd_batch.exe $rate $lambda $nItr $thds $grain $dim $avx >> $path/mf_sgd_batch-$mbind-$rate-$lambda-$nItr-$thds-$grain-$dim-$avx.res

# done

