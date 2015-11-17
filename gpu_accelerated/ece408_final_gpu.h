/*  Project:                ECE 408 Final Project
 *  File Name:              ece408_final_gpu.h
 *  Depends on:             utility.cpp kernel.cu
 *  Included by:            main.cpp
 *  Date created:           Sat Nov 7 2015
 *  Engineers:              Peter Iordanov, Laura Galbraith, Conor Gardner
 *  Compiler:               nvcc
 *  Target OS:              Ubuntu Linux 14.04
 *  Target architecture:    CPU: x86 (64 bit)       GPU: Nvida GTX 970
 *  Description:            Function prototypes for CPU version of Conway's Game of Life. */
 
#ifndef __HEADER_GUARD_ECE408_FINAL__
#define __HEADER_GUARD_ECE408_FINAL__

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32
#define HALO 1

/* utility.cpp */
double get_timestamp();

// simple class for generating output filenames
class output_filename_t
{
    public:
        
        char* str;
        
        // constructors, assignment, destructor
        output_filename_t(const char* output_pattern, unsigned iterations);
        output_filename_t(const output_filename_t& rhs);
        output_filename_t& operator=(const output_filename_t& rhs);
        ~output_filename_t();
        
        bool next_filename();
        
    private:
        
        unsigned digit_end;  // 9 if str = "output_042.png", for example
        unsigned len;        // 14 if str = "output_042.png", for example
};

/* kernel.cu */
__global__ void kernel
(
    const unsigned char* read_grid_d,
    unsigned char* write_grid_d,
    unsigned width,
    unsigned height
);

#endif // header guard

