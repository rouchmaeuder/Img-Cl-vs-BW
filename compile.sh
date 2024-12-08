#!/bin/bash

echo "compiling cuda library with nvcc"
/usr/local/cuda/bin/nvcc /home/user/tiff_file_parser/libs/cuda_acceleration.cu -Xcompiler "-fPIC" -o /home/user/tiff_file_parser/libs/cuda_acceleration.o -c -g

echo "compiling tiff lib with gcc"
gcc libs/tiff.c -lm -o libs/tiff.o -c -g

echo "compiling main with gcc"
gcc main.c -lm -o main.o -c -g

echo "linking"
gcc main.o libs/tiff.o libs/cuda_acceleration.o -lcudart -L/usr/local/cuda/lib64 -lm -lstdc++ -o a.out