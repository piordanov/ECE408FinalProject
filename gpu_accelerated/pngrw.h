#ifndef __PNGRW_HEADER_GUARD__
#define __PNGRW_HEADER_GUARD__

/* pngrw_read_file()
 *
 **** Overview ****
 *
 * Opens a png file and uses it to initialze a 2D memory block containing
 * cells for the game of life simulation.
 *
 **** Inputs ****
 *
 * filename   - Path to a png file to open.  The file may be encoded as a
 *              monochrome png, 256-color paletted png, or RGBA png.  Note
 *              black pixels in the png will be converted to zeros in memory
 *              (cell is dead) and non-zero pixels from the input will be
 *              stored as ones in memory (cell is alive).
 * pad_width  - Padding can be optionally inserted at the end of each row
 *              in the output array such that each row is a muptiple of
 *              'pad_width' bytes.  The default option is 1 (no padding)
 *
 **** Outputs ****
 *
 * cell_grid  - This is the output array which is initialized based on the
 *              contents of the input png file.  See "Memory Layout" 
 *              secton below. The user may pass a pre-existing block
 *              of memory to be initialized or the user may
 *              pass a pointer to a NULL pointer to indicate that 
 *              png_read_file should allocate a new block.  See examples:
 *
 *              //  use pre-existing memory 
 *              unsigned char* mem = new unsigned char[...];
 *              png_read_file("file.png", &mem, ...)
 *
 *              // allocate new memory
 *              unsigned char* mem = NULL;
 *              png_read_file("file.png", &mem, ...)
 *              
 * width      - The number of pixels across the x-dimension of the input.
 *              image.  Undefined if read failed.
 * height     - The number of pixels down the y-dimension of the input image.
 *              Undefined if read failed.
 *
 **** Return Value ****
 *
 *            - Zero if an error of any kind occurred.
 *            - The total number bytes in the output array, including padding (if any)
 *
 **** Side Effects ****
 *
 *            - May print to sdtout if an error occurs
 *            - Opens, reads, and closes the file specified by the 'filename' argument
 *
 **** Memory Layout ****
 *
 * Each cell in the grid is tightly packed into a 2D array of bytes.  Each
 * element in the 'cell_grid' array contains the live/dead status of 8 
 * adjacent cells.  The least significant bit of a given byte represents the
 * leftmost cell for that byte.
 *
 * If the 'pad_width' argument is larger than 1, then additional unused bytes
 * may be added to the end of each row to ensure that the number of bytes
 * across each row is divisible by 'pad_width'.  This makes array reads/writes
 * more cache friendly.
 *
 * Please refer to memory_layout.png for an intuitive picture */

unsigned pngrw_read_file
(
    // outputs
    unsigned char** cell_grid,
    unsigned*       width,
    unsigned*       height,
    
    // inputs
    const char*     filename,
    unsigned        pad_width = 1
);

/* pngrw_write_file()
 *
 **** Overview ****
 *
 * Saves a grid of cells to a png file using the same format as pngrw_read_file() 
 *
 **** Inputs ****
 *
 * filename   - Path to a png file to write to.  File will be saved as a monochrome image
 *              with 1 bit per pixel.
 * cell_grid  - This is the input array which is written to an output png
 *              See "Memory Layout" secton below.
 * width      - The number of pixels across the x-dimension of the input.
 *              image.  Undefined if read failed.
 * height     - The number of pixels down the y-dimension of the input image.
 *              Undefined if read failed.
 * pad_width  - If padding was used for the array, it must me specified here.
 *              The default option is 1 (no padding)
 *
 **** No Outputs ****
 *
 **** Return Value ****
 *
 *            - true on success
 *            - false on failure
 *
 **** Side Effects ****
 *
 *            - May print to sdtout if an error occurs
 *            - Opens, writes, and closes the file specified by the 'filename' argument
 *
 **** Memory Layout ****
 *
 * Each cell in the grid is tightly packed into a 2D array of bytes.  Each
 * element in the 'cell_grid' array contains the live/dead status of 8 
 * adjacent cells.  The least significant bit of a given byte represents the
 * leftmost cell for that byte.  Bits are 1 for live cells and 0 for dead cells.
 *
 * Examples:
 *  cell_grid[0] & 0x1 is the cell in the upper-left corner
 *  (cell_grid[0] & 0x2) >> 1 is the next cell to the right
 *  cell_grid[1] & 0x1 is the 8th cell to the right of the upper left corner
 * 
 * If the 'pad_width' argument is larger than 1, then additional unused bytes
 * may be added to the end of each row to ensure that the number of bytes
 * across each row is divisible by 'pad_width'.  This makes array reads/writes
 * more cache friendly.
 *
 * Please refer to memory_layout.png for an intuitive picture */

bool pngrw_write_file
(
    const char* filename,
    const unsigned char* cell_grid,
    unsigned width,
    unsigned height,
    unsigned pad_width = 1
);

#endif

