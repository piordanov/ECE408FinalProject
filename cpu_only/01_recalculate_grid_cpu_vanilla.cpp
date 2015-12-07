/*  Project:                ECE 408 Final Project
 *  File Name:              recalculate_grid_cpu_01_vanilla.cpp
 *  Calls:                  none
 *  Called by:              main.cpp
 *  Associated Header:      ece408_final_cpu.h
 *  Date created:           Sat Nov 7 2015
 *  Engineers:              Peter Iordanov, Laura Galbraith, Conor Gardner
 *  Compiler:               g++
 *  Target OS:              Ubuntu Linux 14.04
 *  Target architecture:    x86 (64 bit)
 *  Description:            Reads an input cell grid, computes the next generation (step)
 *                          and writes it to the output grid */

#include <cstring>
#include <cstdio>

bool get_cell
(
    const unsigned char* input_cell_grid,
    int x,
    int y,
    unsigned width,
    unsigned height
){
    // Simply return if the cell at the given (x,y) coordinate is alive
    // Check that x,y are inside bounds of grid
    if (x >= (int)width || y >= (int)height || x < 0 || y < 0)
        return false;

    unsigned int remainder = (unsigned int)(x % 8);
    unsigned char mask = 0x1 << remainder;
    unsigned char cell_char = input_cell_grid[y * ((width-1)/8 +1) + x/8];
    unsigned char masked_char = cell_char & mask;
    return (masked_char != 0);
}

void write_cell
(
    bool alive,
    int x,
    int y,
    unsigned width,
    unsigned height,
    unsigned char* output_cell_grid
){
    
    unsigned char mask = 0x1 << (x % 8);
    unsigned char cell_char = output_cell_grid[y * ((width-1)/8 +1) + x/8];
    if (alive)
        output_cell_grid[y * ((width-1)/8 +1) + x/8] = cell_char | mask;
    else
        output_cell_grid[y * ((width-1)/8 +1) + x/8] = cell_char & ~mask;
    
}

void recalculate_grid_cpu
(
    unsigned char* output_cell_grid,
    const unsigned char* input_cell_grid,
    unsigned width,
    unsigned height
){
    
    // 1) Any live cell with fewer than two live neighbours dies, as if caused by under-population.
    // 2) Any live cell with two or three live neighbours lives on to the next generation.
    // 3) Any live cell with more than three live neighbours dies, as if by over-population.
    // 4) Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
    //
    // and...
    //
    // if a cell is on a boundary, non-existent neighbors should be counted as dead cells.

    // Loop through every x and y, count the number of alive neighbors,
    // apply the rule, fill in output_cell_grid
    
    for (int x = 0; x < (int)width; x++) {
        for (int y = 0; y < (int)height; y++) {
            int neighbor_sum = 0;
            bool curr_cell_alive = get_cell(input_cell_grid, x, y, width, height);
            bool next_cell_alive = curr_cell_alive;
            //get sum of neighbors
            for(int j = -1; j < 2; j++) {
                for(int i = -1; i < 2; i++) {
                    if (i != 0 || j != 0)
                    {
                        if (get_cell(input_cell_grid, x+i, y+j, width, height)) {
                            neighbor_sum++;
                        }
                    }    
                }
            }
            //check rules
            if(curr_cell_alive)
            {
                if(neighbor_sum < 2)//under-population rule
                    next_cell_alive = false;
                else if(neighbor_sum < 4)
                    next_cell_alive = true;
                else
                    next_cell_alive = false;
            }
            else
            {
                if(neighbor_sum == 3)
                    next_cell_alive = true;
                else
                    next_cell_alive = false;
            }
            write_cell(next_cell_alive, x, y, width, height, output_cell_grid);
        }
    }
    
}
