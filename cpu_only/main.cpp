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
 *                              ./life_cpu [initial_grid.png] [iterations] */

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <locale.h>
#include "pngrw.h"
#include "ece408_final_cpu.h"

unsigned first_filename(unsigned iterations, char** output_filename);
void next_filename(unsigned digit, char* output_filename);

int main(int argc, char** argv)
{
    
    // abort if the user didn't pass the correct number of command line args
    if (argc != 3)
    {
        
        fprintf
        (
            stderr,
            "Error! Invalid command line arguments.\n"
            "Usage: %s [initial_grid.png] [iterations]\n",
            argv[0]
        );
        
        return -1;
        
    } else if (argc > 3) {
        
        fprintf
        (
            stderr,
            "Warning! Ignoring all but the first 2 command line arguments\n"
            "Usage: %s [initial_grid.png] [iterations]\n",
            argv[0]
        );
        
    }
    
    // detect the number of iterations
    int iterations = atoi(argv[2]);
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
    
    // we're double buffering, pngrw_read_file only allocated the input buffer
    cell_grid_write = new unsigned char[bytes];
    
    // simulate the cell grid for 'iterations' steps
    char* output_filename;
    unsigned digit = first_filename(iterations, &output_filename);
    
    double compute_elapsed_sec = 0.0;
    
    printf("first (live) = %u, second (dead) = %u\n", cell_grid_read[0] & 0x1, (cell_grid_read[0] & 0x2) >> 1);
    
    for (int ix = 0; ix < iterations; ix++)
    {
        
        // simulate one time step (generation)
        double start = get_timestamp();
        recalculate_grid_cpu(cell_grid_write, cell_grid_read, width, height);
        compute_elapsed_sec += get_timestamp() - start;
        
        // write the new cell grid to a png file
        pngrw_write_file
        (
            output_filename,
            cell_grid_write,
            width,
            height
        );
        
        next_filename(digit, output_filename);
        
        // swap buffers
        unsigned char* swap = cell_grid_read;
        cell_grid_read = cell_grid_write;
        cell_grid_write = swap;
        
    }
    
    // cleanup
    delete[] output_filename;
    delete[] cell_grid_read;
    delete[] cell_grid_write;
    
    setlocale(LC_NUMERIC, ""); // for thousands separator
    printf
    (
        "Success! Finished %u iterations in %lf seconds (%'u cells/sec)\n",
        iterations,
        compute_elapsed_sec,
        (unsigned)((double)(width * height * iterations) / compute_elapsed_sec)
    );
    
    return 0;
    
}

// creates filenames such as generation_00.png for iteration counts between 10 and 99
// generation_000.png for 100 to 999 etc
// returns the index of the 1's digit in the generated string
unsigned first_filename(unsigned iterations, char** output_filename)
{
    
    unsigned digits = 0;
    while (iterations != 0)
    {
        digits++;
        iterations /= 10;
    }
    
    *output_filename = new char[digits + 16];
    
    memcpy(*output_filename, "generation_", 11 * sizeof(char));
    memset(*output_filename + 11, '0', digits * sizeof(char));
    memcpy(*output_filename + 11 + digits, ".png", 5);
    
    return digits + 10;
    
}

// increments the numerical part of the output filename
void next_filename(unsigned digit, char* output_filename)
{
    while (true)
    {
        char fetch = output_filename[digit];
        fetch++;
        if (fetch > '9')
        {
            fetch = '0';
            output_filename[digit] = fetch;
            digit--;
        } else {
            output_filename[digit] = fetch;
            return;
        }
    }
}

