/*  Project:                ECE 408 Final Project
 *  File Name:              main.cpp
 *  Calls:                  pngrw.cpp recalculate_grid_cpu.cpp
 *  Called by:              None
 *  Associated Header:      ece408_final_cpu.h
 *  Date created:           Sat Nov 7 2015
 *  Engineers:              Conor Gardner
 *  Compiler:               g++
 *  Target OS:              Ubuntu Linux 14.04
 *  Target architecture:    x86 (64 bit)
 *  Description:            Simulates Conway's Game of Life.  Optionally loads an initial
 *                          grid of cells and iterates through the requested number of steps.
 *                          This is the simple vanilla version.  Minimal optimizations, single
 *                          threaded
 *
 *                          Valid Command Line Usages:
 *                              ./life_cpu [initial_grid.png] [output_pattern] [iterations] */

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <locale.h>
#include "pngrw.h"
#include "ece408_final_cpu.h"

int main(int argc, char** argv)
{
    
    // abort if the user didn't pass the correct number of command line args
    if (argc != 4)
    {
        
        fprintf
        (
            stderr,
            "Error! Invalid command line arguments.\n"
            "Usage: %s [initial_grid.png] [output_pattern] [iterations]\n",
            argv[0]
        );
        
        return -1;
        
    } else if (argc > 4) {
        
        fprintf
        (
            stderr,
            "Warning! Ignoring all but the first 3 command line arguments\n"
            "Usage: %s [initial_grid.png] [output_pattern] [iterations]\n",
            argv[0]
        );
        
    }
    
    // detect the number of iterations
    unsigned iterations = atoi(argv[3]);
    if (iterations < 1)
    {
        fprintf(stderr, "Error! Iteration count must be at least 1\n");
        return -1;
    }
    
    unsigned width;
    unsigned height;
    
    // double buffered cell memory.
    // cell_grid_read is NULL to tell pngrw_read_file() to allocate new memory for us
    unsigned char* cell_grid_read = NULL;
    unsigned char* cell_grid_write;
    
    // attempt to load the inital cell grid.  cell_grid_read will be allocated
    unsigned bytes = pngrw_read_file(&cell_grid_read, &width, &height, argv[1]);
    if (bytes == 0)
        // error messages printed from inside pngrw_read_file()
        return -1;
    
    if (width < 2)
    {
        fprintf(stderr, "Error! %s must have a width > 1\n", argv[1]);
        return -1;
    }
    
    if (height < 2)
    {
        fprintf(stderr, "Error! %s must have a height > 1\n", argv[1]);
        return -1;
    }
    
    // we're double buffering, pngrw_read_file only allocated the input buffer
    cell_grid_write = new unsigned char[bytes];
    // need to init to 0 so valgrind won't complain (it can't track single-bit initializations)
    memset(cell_grid_write, 0, bytes);
    
    // simulate the cell grid for 'iterations' steps
    output_filename_t output_filename(argv[2], iterations);
    
    double start = get_timestamp();
    for (unsigned ix = 0; ix < iterations; ix++)
    {
        
        // simulate one time step (generation)
        recalculate_grid_cpu(cell_grid_write, cell_grid_read, width, height);
        
        // write the new cell grid to a png file
        pngrw_write_file
        (
            output_filename.str,
            cell_grid_write,
            width,
            height
        );
        
        output_filename.next_filename();
        
        // swap buffers
        unsigned char* swap = cell_grid_read;
        cell_grid_read = cell_grid_write;
        cell_grid_write = swap;
        
    }
    double elapsed_sec = get_timestamp() - start;
    
    // cleanup
    delete[] cell_grid_read;
    delete[] cell_grid_write;
    
    printf
    (
        "Success! Finished %u iterations in %lf seconds (%lf cells/sec)\n",
        iterations,
        elapsed_sec,
        ((double)width * (double)height * (double)iterations) / elapsed_sec
    );
    
    return 0;
    
}

