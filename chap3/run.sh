# /bin/bash

cd ~/cuda-learn/chap3
nvcc reduceInteger.cu
nvprof --metrics inst_per_warp ./a.out
nvprof --metrics gld_throughput ./a.out
nvprof --metrics gld_efficiency ./a.out