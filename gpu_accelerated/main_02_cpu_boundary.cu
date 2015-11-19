/*  Project:                ECE 408 Final Project
 *  File Name:              main_02_cpu_boundary.cu
 *  Calls:                  pngrw.cpp
 *  Called by:              none
 *  Associated Header:      ece408_final_gpu.h
 *  Date created:           Tues Nov 17 2015
 *  Engineers:              Peter Iordanov, Laura Galbraith, Conor Gardner
 *  Compiler:               nvcc
 *  Target OS:              Ubuntu Linux 14.04
 *  Target architecture:    CPU: x86 (64 bit)       GPU: Nvida GTX 970 */

#include <cstdio>
#include "ece408_final_gpu.h"
#include "pngrw.h"

const unsigned char bit_count[256] = 
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

inline bool check_rule (unsigned packed_neighbors, bool old)
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
inline unsigned char generate_byte
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

// Calculate and write ONLY the boundary row/columns of cells into write_grid_h
void recalculate_boundary_cells
(
    const unsigned char* read_grid_h,
    unsigned char* write_grid_h,
    unsigned width,
    unsigned height
){

    unsigned bytes_per_row = (width - 1) / 8 + 1;

    unsigned char nw, n, ne;
    unsigned char w,  c, e;
    unsigned char sw, s, se;

    const unsigned char* w_char, *nw_char, *sw_char;

    // compute and write the north row
    for (unsigned ix = 0, iy = 0; ix < bytes_per_row; ix++)
    {
        w_char = read_grid_h + (iy * bytes_per_row) + ix - 1;
        sw_char = w_char + bytes_per_row;
        if (ix == 0)
            w = 0x0;
        else
            w = w_char[0];
        c = w_char[1];
        if (ix == bytes_per_row - 1)
            e = 0x0;
        else
            e = w_char[2];
        if (ix == 0 || iy == height - 1)
            sw = 0x0;
        else
            sw = sw_char[0];
        if (iy == height - 1)
            s = 0x0;
        else
            s = sw_char[1];
        if (ix == bytes_per_row - 1 || iy == height - 1)
            se = 0x0;
        else
            se = sw_char[2];

        write_grid_h[iy * bytes_per_row + ix] = generate_byte(0x0, 0x0, 0x0, w, c, e, sw, s, se);
    }

    // compute the south row and make sure it is not equal to the north row
    for (unsigned ix = 0, iy = height - 1; ix < bytes_per_row && iy != 0; ix++)
    {
        w_char = read_grid_h + (iy * bytes_per_row) + ix - 1;
        nw_char = w_char - bytes_per_row;
        if (ix == 0)
            nw = 0x0;
        else
            nw = nw_char[0];
        n = nw_char[1];
        if (ix == bytes_per_row - 1)
            ne = 0x0;
        else
            ne = nw_char[2];
        if (ix == 0)
            w = 0x0;
        else
            w = w_char[0];
        c = w_char[1];
        if (ix == bytes_per_row - 1)
            e = 0x0;
        else
            e = w_char[2];

        write_grid_h[iy * bytes_per_row + ix] = generate_byte(nw, n, ne, w, c, e, 0x0, 0x0, 0x0);
    }

    // compute the west column and skip the (0,0),(0,height-1) chars
    for (unsigned ix = 0, iy = 1; iy < height - 1; iy++)
    {
        w_char = read_grid_h + (iy * bytes_per_row) + ix - 1;
        nw_char = w_char - bytes_per_row;
        sw_char = w_char + bytes_per_row;
        n = nw_char[1];
        if (ix == bytes_per_row - 1)
            ne = 0x0;
        else
            ne = nw_char[2];
        c = w_char[1];
        if (ix == bytes_per_row - 1)
            e = 0x0;
        else
            e = w_char[2];
        s = sw_char[1];
        se = sw_char[2];

        write_grid_h[iy * bytes_per_row + ix] = generate_byte(0x0, n, ne, 0x0, c, e, 0x0, s, se);
    }

    // compute the east column and skip the (bytes_per_row-1,0),(bytes_per_row-1,height-1) chars
    // also make sure it is not equal to the west column
    for (unsigned ix = bytes_per_row - 1, iy = 1; ix != 0 && iy < height - 1; iy++)
    {
        w_char = read_grid_h + (iy * bytes_per_row) + ix - 1;
        nw_char = w_char - bytes_per_row;
        sw_char = w_char + bytes_per_row;
        nw = nw_char[0];
        n = nw_char[1];
        w = w_char[0];
        c = w_char[1];
        sw = sw_char[0];
        s = sw_char[1];

        write_grid_h[iy * bytes_per_row + ix] = generate_byte(nw, n, 0x0, w, c, 0x0, sw, s, 0x0);
    }

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
    
    // blocking copy initial grid to GPU
    cudaMemcpy(read_grid_d, send_grid_h, grid_bytes * sizeof(char), cudaMemcpyHostToDevice);

    /* Stage 3 - Simulate each generation of cells on the GPU */
    
    // allocate page-locked memory to recieve frames from the GPU
    unsigned char* recv_grid_h;
    cudaMallocHost(&recv_grid_h, grid_bytes * sizeof(char));
    
    // generates a name such as output_000.png on the heap
    // if argv[2] = "output" and 99 < iterations < 1000, for example
    output_filename_t output_filename(argv[2], iterations);
    
    // use the GPU to simulate all iterations but only for the non-boundary body of the grid
    dim3 threads_per_block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
    unsigned body_chars_x = (width - 1) / 8 - 1;
    unsigned body_chars_y = height - 2;
    dim3 blocks_per_grid
    (
        (body_chars_x - 1) / THREADS_PER_BLOCK_X + 1,
        (body_chars_y - 1) / THREADS_PER_BLOCK_Y + 1,
        1
    );
    for (unsigned gen_ix = 0; gen_ix < iterations; gen_ix++)
    {
        
        output_filename.next_filename();
        // Check that we need to run the GPU on this grid at all
        if (blocks_per_grid.x > 0 && blocks_per_grid.y > 0)
        {
            kernel<<<blocks_per_grid, threads_per_block>>>(read_grid_d, write_grid_d, width, height);
            cudaMemcpy(recv_grid_h, write_grid_d, grid_bytes * sizeof(char), cudaMemcpyDeviceToHost);
        }

        // do boundary calculations using send_grid_h into recv_grid_h
        recalculate_boundary_cells(send_grid_h, recv_grid_h, width, height);

        pngrw_write_file(output_filename.str, recv_grid_h, width, height);

        // copy recv_grid_h with newly-computed boundaries into the inputs of the next iteration
        memcpy(send_grid_h, recv_grid_h, grid_bytes * sizeof(char));
        if (blocks_per_grid.x > 0 && blocks_per_grid.y > 0)
        {
            cudaMemcpy(write_grid_d, recv_grid_h, grid_bytes * sizeof(char), cudaMemcpyHostToDevice);
        
            unsigned char* swap = read_grid_d;
            read_grid_d = write_grid_d;
            write_grid_d = swap;
        }

    }
    
    /* Stage 4 - cleanup */
    
    // cleanup
    cudaFreeHost(recv_grid_h);
    cudaFree(read_grid_d);
    cudaFree(write_grid_d);

    // we're done with the CPU's copy of the input grid
    delete[] send_grid_h;
    
    return 0;
    
}

