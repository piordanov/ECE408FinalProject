/*  Project:                ECE 408 Final Project
 *  File Name:              kernel.cu
 *  Calls:                  none
 *  Called by:              main.cpp
 *  Associated Header:      ece408_final_gpu.h
 *  Date created:           Wed Nov 11 2015
 *  Engineers:              Peter Iordanov, Laura Galbraith, Conor Gardner
 *  Compiler:               nvcc
 *  Target OS:              Ubuntu Linux 14.04
 *  Target architecture:    CPU: x86 (64 bit)       GPU: Nvida GTX 970 */

#include "ece408_final_gpu.h"

__global__ void kernel
(
    const unsigned char* read_grid_d,
    unsigned char* write_grid_d,
    unsigned width,
    unsigned height
){
    
    // TODO compute the next generation of cells on the GPU.
    // Right now it just copies the memory as a self-test
    unsigned bytes_per_row = (width - 1) / 8 + 1;
    
    unsigned ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned iy = (blockIdx.y * blockDim.y) + threadIdx.y;
    
    // TODO you can delete the 5 lines below when you write the actual kernel code
    if (ix < bytes_per_row && iy < height)
    {
        unsigned flat_ix = iy * bytes_per_row + ix;
        write_grid_d[flat_ix] = read_grid_d[flat_ix]; 
    }
    
}

