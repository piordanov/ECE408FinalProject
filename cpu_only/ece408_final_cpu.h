/*  Project:                ECE 408 Final Project
 *  File Name:              ece408_final_cpu.h
 *  Depends on:             None
 *  Included by:            main.cpp
 *  Date created:           Sat Nov 7 2015
 *  Engineers:              Peter Iordanov, Laura Galbraith, Conor Gardner
 *  Compiler:               g++
 *  Target OS:              Ubuntu Linux 14.04
 *  Target architecture:    x86 64-bit
 *  Description:            Function prototypes for CPU version of Conway's Game of Life. */
 
#ifndef __HEADER_GUARD_ECE408_FINAL__
#define __HEADER_GUARD_ECE408_FINAL__

/*** Function Name ***
 *  main
 *
 *** Function Description ***
 *      Launches a Game of Life simulation by loading an intial cell grid from a png file
 *  and simulating a specified number of generations.  Each generation is written to an
 *  output png file
 *
 *** Inputs ****
 *  argc    -   The number of arguments passed to main() and the number of strings in argv
 *  argv[1] -   Path to a png file which specifies the inital cell grid
 *  argv[2] -   The number of iterations to simulate.  Must be at least 1
 *
 *** Outputs ***
 *  none
 *
 *** Returns ***
 *  0       if program completed successfully
 *  else    if program encountered an error
 *
 *** Assumptions - Function is allowed to crash if any of the following aren't met ***
 *  None - Function must behave correctly under all conditions.
 *
 *** Side Effects ***
 *  None */
int main(int argc, char** argv);

/*** Function Name ***
 *  recalculate_grid_cpu
 *
 *** Function Description ***
 *      TODO - Laura or Peter
 *
 *** Inputs ****
 *      input_cell_grid TODO - Laura or Peter
 *      width   - The number of cells across a row of the cell grid.  8 cells = 1 byte
 *      height  - The number of cells down a column of a cell grid.
 *      
 *** Outputs ***
 *  output_cell_grid TODO - Laura or Peter
 *
 *** Assumptions - Function is allowed to crash if any of the following aren't met ***
 *  TODO - Laura or Peter
 *
 *** Side Effects ***
 *  TODO - Laura or Peter */
void recalculate_grid_cpu
(
    unsigned char* output_cell_grid,
    const unsigned char* input_cell_grid,
    unsigned width,
    unsigned height
);

/* Returns the number of seconds since some unknown epoch with microsecond precision */
double get_timestamp();

#endif // header guard
