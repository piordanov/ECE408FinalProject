/*  Project:                ECE 408 Final Project
 *  File Name:              kernel.cu
 *  Calls:                  none
 *  Called by:              main.cpp
 *  Associated Header:      ece408_final_gpu.h
 *  Date created:           Mon Nov 16 2015
 *  Engineers:              Peter Iordanov, Laura Galbraith, Conor Gardner
 *  Compiler:               nvcc
 *  Target OS:              Ubuntu Linux 14.04
 *  Target architecture:    CPU: x86 (64 bit)       GPU: Nvida GTX 970 */

#include "ece408_final_gpu.h"

__constant__ unsigned char bit_count[256] = 
{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};

__device__ inline bool check_rule (unsigned packed_neighbors, bool old)
{
    unsigned neighbor_count = bit_count[packed_neighbors];
    
    // 1) Any live cell with fewer than two live neighbours dies, as if caused by under-population.
    // 2) Any live cell with two or three live neighbours lives on to the next generation.
    // 3) Any live cell with more than three live neighbours dies, as if by over-population.
    // 4) Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
    
    if (old)
        return neighbor_count == 2 || neighbor_count == 3;
    else
        return neighbor_count == 3;

}

// Takes in a center byte and its neighbors, and computes the next center byte
__device__ inline unsigned char generate_byte
(
    unsigned char nw,   unsigned char n,    unsigned char ne,
    unsigned char w,    unsigned char c,    unsigned char e,
    unsigned char sw,   unsigned char s,    unsigned char se
){
    
    unsigned char new_output = 0x00;
    
    // output bit 0
    unsigned char packed_neighbors =
        (nw & 0x80) | ((n & 0x03) << 5)
    |   ((w & 0x80) >> 3) | ((c & 0x02) << 2)
    |   ((sw & 0x80) >> 5) | (s & 0x03);
    if (check_rule(packed_neighbors, c & 0x01))
        new_output |= 0x01;
    
    // output bit 1
    packed_neighbors =
        ((n & 0x07) << 5)
    |   ((c & 0x01) << 4) | ((c & 0x04) << 1)
    |   ((s & 0x07));
    if (check_rule(packed_neighbors, c & 0x02))
        new_output |= 0x02;
    
    // output bit 2
    packed_neighbors = 
        ((n & 0x0E) << 4)
    |   ((c & 0x02) << 3) | (c & 0x08)
    |   ((s & 0x0E) >> 1);
    if (check_rule(packed_neighbors, c & 0x04))
        new_output |= 0x04;
    
    // output bit 3
    packed_neighbors = 
        ((n & 0x1C) << 3)
    |   ((c & 0x04) << 2) | ((c & 0x10) >> 1)
    |   ((s & 0x1C) >> 2);
    if (check_rule(packed_neighbors, c & 0x08))
        new_output |= 0x08;
    
    // output bit 4
    packed_neighbors =
        ((n & 0x38) << 2)
    |   ((c & 0x08) << 1) | ((c & 0x20) >> 2)
    |   ((s & 0x38) >> 3);
    if (check_rule(packed_neighbors, c & 0x10))
        new_output |= 0x10;
    
    // output bit 5
    packed_neighbors =
        ((n & 0x70) << 1)
    |   (c & 0x10) | ((c & 0x40) >> 3)
    |   ((s & 0x70) >> 4);
    if (check_rule(packed_neighbors, c & 0x20))
        new_output |= 0x20;
    
    // output bit 6
    packed_neighbors =
        (n & 0xE0)
    |   ((c & 0x20) >> 1) | ((c & 0x80) >> 4)
    |   ((s & 0xE0) >> 5);
    if (check_rule(packed_neighbors, c & 0x40))
        new_output |= 0x40;
    
    // output bit 7
    packed_neighbors =
        (n & 0xC0) | ((ne & 0x01) << 5)
    |   ((c & 0x40) >> 2) | ((e & 0x01) << 3)
    |   ((s & 0xC0) >> 5) | (se & 0x01);
    if (check_rule(packed_neighbors, c & 0x80))
        new_output |= 0x80;
    
    return new_output;
    
}

// Each thread will calculate 8 cells (1 byte) of the next generation
__global__ void kernel
(
    const unsigned char* read_grid_d,
    unsigned char* write_grid_d,
    unsigned width,
    unsigned height
){

    unsigned bytes_per_row = (width - 1) / 8 + 1;
    
    // This is the (x,y) coordinate of the char in the grid this thread handles
    unsigned ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned iy = (blockIdx.y * blockDim.y) + threadIdx.y;

    unsigned char nw, n, ne;
    unsigned char w,  c, e;
    unsigned char sw, s, se;

    const unsigned char* w_char = read_grid_d + (iy * bytes_per_row) + ix - 1;
    const unsigned char* nw_char = w_char - bytes_per_row;
    const unsigned char* sw_char = w_char + bytes_per_row;

    if (ix == 0 || iy == 0 || ix > bytes_per_row || iy > height)
        nw = 0x0;
    else
        nw = nw_char[0];

    if (iy == 0 || ix >= bytes_per_row || iy > height) // ix can't be <0
        n = 0x0;
    else
        n = nw_char[1];

    if (iy == 0 || ix >= bytes_per_row - 1 || iy > height) // ix can't be <0
        ne = 0x0;
    else
        ne = nw_char[2];

    if (ix == 0 || ix > bytes_per_row || iy >= height) // iy can't be <0
        w = 0x0;
    else
        w = w_char[0];

    // This check is just to make sure we don't read invalid memory
    if (ix >= bytes_per_row || iy >= height) // ix and iy can't be <0
        c = 0x0;
    else
        c = w_char[1];

    if (ix >= bytes_per_row - 1 || iy >= height) // ix and iy can't be <0
        e = 0x0;
    else
        e = w_char[2];

    if (ix == 0 || ix > bytes_per_row || iy >= height - 1) // iy can't be <0
        sw = 0x0;
    else
        sw = sw_char[0];

    if (ix >= bytes_per_row || iy >= height - 1) // ix and iy can't be <0
        s = 0x0;
    else
        s = sw_char[1];

    if (ix >= bytes_per_row - 1 || iy >= height - 1)  // ix and iy can't be <0
        se = 0x0;
    else
        se = sw_char[2];

    // Compute and write the output char
    if (ix < bytes_per_row && iy < height)
    {
        write_grid_d[iy * bytes_per_row + ix]
                    = generate_byte(nw, n, ne, w, c, e, sw, s, se);
    }

}

