This projet showcase the difference in terms of accuracy for SpMV operation based on the CUDA version.

# Compiling & running

nvcc compare_cpu_gpu_spmv.cu -lcusparse
./a.out

# Results

## cuSparse version = 12.1

L2 difference between CPU and custom GPU results: 11.8698

Number of values with absolute difference > 0.1: 1036

## cuSparse version = 11.3

L2 difference between CPU and custom GPU results: 8.27687e-16

Number of values with absolute difference > 0.1: 0