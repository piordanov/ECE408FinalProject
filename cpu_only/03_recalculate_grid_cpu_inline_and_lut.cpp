/*  Project:                ECE 408 Final Project
 *  File Name:              recalculate_grid_cpu_03_inline_and_lut.cpp
 *  Calls:                  none
 *  Called by:              main.cpp
 *  Associated Header:      ece408_final_cpu.h
 *  Date created:           Sat Nov 14 2015
 *  Engineers:              Peter Iordanov, Laura Galbraith, Conor Gardner
 *  Compiler:               g++
 *  Target OS:              Ubuntu Linux 14.04
 *  Target architecture:    x86 (64 bit)
 *  Description:            Reads an input cell grid, computes the next generation (step)
 *                          and writes it to the output grid */

#include <cstring>
#include <cstdio>

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

inline bool check_rule(unsigned packed_neighbors, bool old)
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

// takes in 9 bytes - a center byte and its neighbors and computes next center byte
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

void recalculate_grid_cpu
(
    unsigned char* output_cell_grid,
    const unsigned char* input_cell_grid,
    unsigned width,
    unsigned height
){
    
    unsigned bytes_per_row = (width - 1) / 8 + 1;
    
    unsigned char nw, n, ne;
    unsigned char w,  c, e;
    unsigned char sw, s, se;
    
    /* Handle awkward edge cases explicitly */
    
    // edge case 1 - cell grid has a zero dimension
    if (width < 1)
    {
        fprintf(stderr, "Warning! recalculate_grid_cpu() called with width = 0\n");
        return;
    }
    if (height < 1)
    {
        fprintf(stderr, "Warning! recalculate_grid_cpu() called with height = 0\n");
        return;
    }
    
    // used to zero-out the east edge of the grid when the byte is only 
    // partially filled
    unsigned east_mask = 0xFF >> ((8 - (width & 0x7)) & 0x7);
    
    // edge case 2 - cell grid is only one byte
    if (bytes_per_row == 1 && height == 1)
    {
        output_cell_grid[0] = east_mask & generate_byte
        (
            0, 0, 0,
            0, input_cell_grid[0], 0,
            0, 0, 0
        );
        return;
    }
    
    // edge case 3 - cell grid is a vertical column where height >= 2
    if (bytes_per_row == 1)
    {
        output_cell_grid[0] = generate_byte
        (
            0, 0, 0,
            0, input_cell_grid[0], 0,
            0, input_cell_grid[1], 0
        );
        
        printf("\t[%u]\t%x --> %x\n", 0, input_cell_grid[0], output_cell_grid[0]);
        for (unsigned iy = 1, bound = height - 1; iy < bound; iy++)
        {
            output_cell_grid[iy] = east_mask & generate_byte
            (
                0, input_cell_grid[iy - 1], 0,
                0, input_cell_grid[iy], 0,
                0, input_cell_grid[iy + 1], 0
            );
            printf("\t[%u]\t%x --> %x\n", iy, input_cell_grid[iy], output_cell_grid[iy]);
        }
        
        output_cell_grid[height - 1] = generate_byte
        (
            0, input_cell_grid[height - 2], 0,
            0, input_cell_grid[height - 1], 0,
            0, 0, 0
        );
        printf("\t[%u]\t%x --> %x\n", height - 1, input_cell_grid[height - 1], output_cell_grid[height - 1]);
        
        return;
        
    }
    
    // edge case 4 - cell grid is a horizontal row where bytes_per_row >= 2
    if (height == 1)
    {
        output_cell_grid[0] = generate_byte
        (
            0, 0, 0,
            0, input_cell_grid[0], input_cell_grid[1],
            0, 0, 0
        );
        
        for (unsigned ix = 1, bound = bytes_per_row - 1; ix < bound; ix++)
            output_cell_grid[ix] = generate_byte
            (
                0, 0, 0,
                input_cell_grid[ix - 1], input_cell_grid[ix], input_cell_grid[ix + 1],
                0, 0, 0
            );
        
        output_cell_grid[bytes_per_row - 1] = generate_byte
        (
            0, 0, 0,
            input_cell_grid[bytes_per_row - 2], input_cell_grid[bytes_per_row - 1], 0,
            0, 0, 0
        );
        
        return;
        
    }
    
    // below this line, bytes_per_row >= 2 and height >= 2 guarenteed
    
    /* Compute bytes on the perimeter of the grid */
    
    // compute nw corner
    output_cell_grid[0] = generate_byte
    (
        0, 0, 0,
        0, input_cell_grid[0], input_cell_grid[1],
        0, input_cell_grid[bytes_per_row], input_cell_grid[bytes_per_row + 1]
    );
    
    // compute ne corner
    output_cell_grid[bytes_per_row - 1] = east_mask & generate_byte
    (
        0, 0, 0,
        input_cell_grid[bytes_per_row - 2], input_cell_grid[bytes_per_row - 1], 0,
        0, 0, 0
    );
    
    // compute sw corner
    const unsigned char* input_almost_bottom = input_cell_grid + bytes_per_row * (height - 2);
    unsigned char* output_bottom = output_cell_grid + bytes_per_row * (height - 1);
    // beware: subtracting values from pointers is a bad idea since the pointers
    // are generally 64 bits but the offset isn't always sign extended
    output_bottom[0] = generate_byte
    (
        0, input_almost_bottom[0], input_almost_bottom[1],
        0, input_almost_bottom[bytes_per_row], input_almost_bottom[bytes_per_row + 1],
        0, 0, 0
    );
    
    // compute se corner
    output_bottom[bytes_per_row - 1] = east_mask & generate_byte
    (
        input_almost_bottom[bytes_per_row - 2], input_almost_bottom[bytes_per_row - 1], 0,
        input_almost_bottom[2 * bytes_per_row - 2], input_almost_bottom[2 * bytes_per_row - 1], 0,
        0, 0, 0
    );
    
    // compute north and south edges (no corners)
    for (unsigned ix = 1, bound = bytes_per_row - 1; ix < bound; ix++)
    {
        
        // north edge
        output_cell_grid[ix] = generate_byte
        (
            0,
            0,
            0,
            input_cell_grid[ix - 1],
            input_cell_grid[ix],
            input_cell_grid[ix + 1],
            input_cell_grid[ix + bytes_per_row - 1],
            input_cell_grid[ix + bytes_per_row],
            input_cell_grid[ix + bytes_per_row + 1]
        );
        
        // south edge
        output_bottom[ix] = generate_byte
        (
            input_almost_bottom[ix - 1],
            input_almost_bottom[ix],
            input_almost_bottom[ix + 1],
            input_almost_bottom[ix + bytes_per_row - 1],
            input_almost_bottom[ix + bytes_per_row],
            input_almost_bottom[ix + bytes_per_row + 1],
            0,
            0,
            0
        );
        
    }
    
    // compute west and east edges (no corners)
    for
    (
        unsigned iy = bytes_per_row, bound = bytes_per_row * (height - 1);
        iy < bound;
        iy += bytes_per_row
    ){
        
        // west edge
        output_cell_grid[iy] = generate_byte
        (
            0,
            input_cell_grid[iy - bytes_per_row],
            input_cell_grid[iy - bytes_per_row + 1],
            0,
            input_cell_grid[iy],
            input_cell_grid[iy + 1],
            0,
            input_cell_grid[iy + bytes_per_row],
            input_cell_grid[iy + bytes_per_row + 1]
        );
        
        // east edge
        output_cell_grid[iy + bytes_per_row - 1] = east_mask & generate_byte
        (
            input_cell_grid[iy - 2],
            input_cell_grid[iy - 1],
            0,
            input_cell_grid[iy + bytes_per_row - 2],
            input_cell_grid[iy + bytes_per_row - 1],
            0,
            input_cell_grid[iy + 2 * bytes_per_row - 2],
            input_cell_grid[iy + 2 * bytes_per_row - 1],
            0
        );
        
    }
    
    // compute main body - optimizing out boundary checks ;)
    // this is the only part of the code that's actually interesting
    // ix and iy select the byte being written
    for (unsigned iy = 1, bound_x = bytes_per_row - 1, bound_y = height - 1; iy < bound_y; iy++)
    {
        
        const unsigned char* row_above = input_cell_grid + bytes_per_row * (iy - 1);
        const unsigned char* row_mid = row_above + bytes_per_row;
        const unsigned char* row_below = row_above + 2 * bytes_per_row;
        
        // these will get shifted in the first iteration
        n = row_above[0];   ne = row_above[1];
        c = row_mid[0];     e  = row_mid[1];
        s = row_below[0];   se = row_below[1];
        
        unsigned char* output_row = output_cell_grid + iy * bytes_per_row;
        
        for (unsigned ix = 1; ix < bound_x; ix++)
        {
            
            // move 3x3 window one cell to the right, recycling 6 out of 9 bytes
            nw = n;     n = ne;     ne = row_above[ix + 1];
            w = c;      c = e;      e = row_mid[ix + 1];
            sw = s;     s = se;     se = row_below[ix + 1];
            
            // we worked hard to get this output! write it
            output_row[ix] = generate_byte(nw, n, ne, w, c, e, sw, s, se);
            
        }
        
    }
    
}

