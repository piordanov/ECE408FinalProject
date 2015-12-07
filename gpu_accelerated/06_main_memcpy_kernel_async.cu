/*  Project:                ECE 408 Final Project
 *  File Name:              main.cu
 *  Calls:                  pngrw.cpp
 *  Called by:              none
 *  Associated Header:      ece408_final_gpu.h
 *  Date created:           Wed Nov 18 2015
 *  Engineers:              Peter Iordanov, Laura Galbraith, Conor Gardner
 *  Compiler:               nvcc
 *  Target OS:              Ubuntu Linux 14.04
 *  Target architecture:    CPU: x86 (64 bit)       GPU: Nvida GTX 970 (Compute Capability 5.2) */

#include <cstdio>
#include <omp.h>
#include "ece408_final_gpu.h"
#include "pngrw.h"

#define PNG_THREADS 4

inline void swap_buffers(unsigned char** a, unsigned char** b)
{
    unsigned char* swap = *a;
    *a = *b;
    *b = swap;
}

int main(int argc, char** argv)
{
    
    /* Stage 1 - parse command line arguments */
    
    // abort if 3 arguments were not passed
    if (argc != 4)
    {
        fprintf
        (
            stderr,
            "Error, invalid command line arguments\n"
            "Usage: %s [inital_grid.png] [output_pattern] [iterations]",
            argv[0]
        );
        return -1;
    }
    
    // abort for an invalid iteration count
    unsigned iterations = atoi(argv[3]);
    if (iterations == 0)
    {
        fprintf(stderr, "Error, iteration count must be at least 1\n");
        return -1;
    }
    
    /* Stage 2 - load starting cell grid and copy it to the GPU */
    
    // Black pixels are dead (0).  Else a cell is live (1)
    unsigned char* send_grid_h = NULL; // starting off NULL tells pngrw to allocate memory for us
    unsigned width;     // the number of cells (not bytes) across a row of send_grid
    unsigned height;    // the number of cells down a column of inital_grid
    unsigned grid_bytes = pngrw_read_file(&send_grid_h, &width, &height, argv[1]);
    if (grid_bytes == 0)
        // failed to read png file.  pngrw will print an error message for us
        return -1;

    // allocate double-buffered GPU memory for input and ouput grids
    unsigned char* read_grid_d;
    unsigned char* write_grid_d;
    cudaMalloc(&read_grid_d, grid_bytes * sizeof(char));
    cudaMalloc(&write_grid_d, grid_bytes * sizeof(char));
    
    double start = get_timestamp();
    
    /* Stage 3 - Copy the initial grid to the GPU and simulate the first generation */
    
    dim3 threads_per_block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
    dim3 blocks_per_grid
    (
        (width / 8 - 1) / THREADS_PER_BLOCK_X + 1,
        (height - 1) / THREADS_PER_BLOCK_Y + 1,
        1
    );

    cudaMemcpy(read_grid_d, send_grid_h, grid_bytes * sizeof(char), cudaMemcpyHostToDevice);
    kernel<<<blocks_per_grid, threads_per_block>>>(read_grid_d, write_grid_d, width, height);
    cudaDeviceSynchronize();
    swap_buffers(&read_grid_d, &write_grid_d);
    
    // we're done with the CPU's copy of the input grid
    delete[] send_grid_h;
    
    /* Stage 4 - Simulate all further simulations on the GPU
     * Overlap CPU execution, device --> host memory copies, and png encodings */
    
    // allocate page-locked memory to recieve frames from the GPU
    unsigned char* recv_grid_h;
    cudaMallocHost(&recv_grid_h, grid_bytes * sizeof(char));
    
    // generates a name such as output_000.png on the heap
    // if argv[2] = "output" and 99 < iterations < 1000, for example
    output_filename_t output_filename(argv[2], iterations);
    
    cudaStream_t kernel_stream;
    cudaStream_t memcpy_stream;
    cudaStreamCreate(&kernel_stream);
    cudaStreamCreate(&memcpy_stream);
    
    for (unsigned gen_ix = 2; gen_ix < iterations; gen_ix++)
    {
        
        kernel
            <<<blocks_per_grid, threads_per_block, 0, kernel_stream>>>
            (read_grid_d, write_grid_d, width, height);
        
        // copy result from *previous* kernel launch in parallel with above kernel execution
        cudaMemcpyAsync
        (
            recv_grid_h,
            read_grid_d,
            grid_bytes * sizeof(char),
            cudaMemcpyDeviceToHost,
            memcpy_stream
        );
        
        cudaDeviceSynchronize();
        
        pngrw_write_file(output_filename.str, recv_grid_h, width, height);
        
        swap_buffers(&read_grid_d, &write_grid_d);
        
        output_filename.next_filename();
        
    }
    
    cudaMemcpy
    (
        recv_grid_h,
        read_grid_d,
        grid_bytes * sizeof(char),
        cudaMemcpyDeviceToHost
    );
    pngrw_write_file(output_filename.str, recv_grid_h, width, height);
    
    cudaStreamDestroy(kernel_stream);
    cudaStreamDestroy(memcpy_stream);
    
    double elapsed_sec = get_timestamp() - start;
    
    /* Stage 5 - cleanup */
    
    // cleanup
    cudaFreeHost(recv_grid_h);
    cudaFree(read_grid_d);
    cudaFree(write_grid_d);
    
    printf
    (
        "Success! Finished %s ~ %u iterations in %lf seconds ~ %e cells/sec\n",
        argv[1],
        iterations,
        elapsed_sec,
        ((double)width * (double)height * (double)iterations) / elapsed_sec
    );
    
    return 0;
    
}

