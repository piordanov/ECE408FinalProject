/*  Project:                ECE 408 Final Project
 *  File Name:              recalculate_grid_cpu.cpp
 *  Calls:                  none
 *  Called by:              main.cpp
 *  Associated Header:      ece408_final_cpu.h
 *  Date created:           Sat Nov 7 2015
 *  Engineers:              TODO
 *  Compiler:               g++
 *  Target OS:              Ubuntu Linux 14.04
 *  Target architecture:    x86 (64 bit)
 *  Description:            Reads an input cell grid, computes the next generation (step)
 *                          and writes it to the output grid */

#include <cstring>
#include <cmath>  // needed for pow

void recalculate_grid_cpu
(
    unsigned char* output_cell_grid,
    const unsigned char* input_cell_grid,
    unsigned width,
    unsigned height
){
    
    // TODO - Laura or Peter.  Compute the next generation of cells (one step)
    // if you don't like the function prototype, feel free to change it.
    // You can erase these comments when you're done, they're mostly to get you started
    //
    // Let's stick to the standard rules (quoted from wikipedia)
    // 1) Any live cell with fewer than two live neighbours dies, as if caused by under-population.
    // 2) Any live cell with two or three live neighbours lives on to the next generation.
    // 3) Any live cell with more than three live neighbours dies, as if by over-population.
    // 4) Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
    //
    // and...
    //
    // if a cell is on a boundary, non-existent neighbors should be counted as dead cells.
    
    // some hints for accessing cells: (see pngrw.h)
    // cell_grid[0] & 0x1 is the cell in the upper-left corner
    // (cell_grid[0] & 0x2) >> 1 is the next cell to the right
    // cell_grid[1] & 0x1 is the 8th cell to the right of the upper left corner
    // Bits are 1 for live cells and 0 for dead cells.
    
    // TODO to test your code, just type make -j valgrind in the terminal.
    // you can look at the output pngs named generation_xx.png to verify that your code works
    // the version of the code that I uploaded functions with no valgrind errors, but that
    // it is still possible that there are bugs in what I've written.  Let me know if you
    // suspect this
    
    // TODO - update "ece408_final_cpu.h" when you finish this function
    // Please update the "Engineers" space at the top of this file with your name.  If someone
    // else has a question, they will know who to ask ;)
    
    // remove the line below, it just copies the input to the output ;)
    // Laura: judging by the memcpy line below, to cover extra (x%8) cells at the end of a
    // line, there is an extra char to hold that info
    // For example, if there are 65 cells in one horizontal line of the grid, there must be
    // 9 chars to hold them; 64 cells->8 chars, 63 cells->8 chars
    // So, to access any cell (x,y): input_cell_grid[y * ((width-1)/8 +1) + x/8] is the right char
    // and 2^(x%8) gives you the correct power-of-2 mask to use on the char
    // If the result is not zero, that means that the cell in question is alive:
    // cell_alive = (input_cell_grid[y*((width-1)/8+1) + x/8] & pow(2.0, (double)(x%8))) != 0;
    memcpy(output_cell_grid, input_cell_grid, ((width - 1) / 8 + 1) * height * sizeof(char));

}
