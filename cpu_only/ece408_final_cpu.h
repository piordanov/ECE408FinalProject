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

/* main.cpp */
int main(int argc, char** argv);

/* recalculate_grid_cpu.cpp */
void recalculate_grid_cpu
(
    unsigned char* output_cell_grid,
    const unsigned char* input_cell_grid,
    unsigned width,
    unsigned height
);

/* utility.cpp */
// get current time in seconds
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

#endif // header guard
