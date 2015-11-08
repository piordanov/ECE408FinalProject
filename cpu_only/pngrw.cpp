/*  Project:                ECE 408 Final Project
 *  File Name:              pngrw.cpp
 *  Calls:                  none
 *  Called by:              main.cpp
 *  Associated Header:      pngrw.h
 *  Date created:           Mon Sep 28 2015
 *  Engineers:              Conor Gardner
 *  Compiler:               g++
 *  Target OS:              Ubuntu Linux 14.04
 *  Target architecture:    x86 (64 bit)
 *  Description:            Reads and writes PNG files and uses them to initialize a cell grid
 *                          for a Conway's Game of Life simulation.  Part of a class project
 *                          (ECE 408) */

#include <png.h> // yucky standard linux png include
#include "pngrw.h" // custom wrapper

/* See pngrw.h for documentation */
unsigned pngrw_read_file
(
    // outputs
    unsigned char** cell_grid,
    unsigned*       width,
    unsigned*       height,
    
    // inputs
    const char*     filename,
    unsigned        pad_width
) {
    
    /* Begin disgusting libpng initialization */
    
    // initialize libpng and bind it to a png file on the filesystem
    png_structp libpng_read_struct;
    png_infop libpng_info_struct;

    // open png file for reading in binary format
    FILE* file = fopen(filename, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "Error (%s) failed to open file for reading\n", filename);
        return false;
    }
    
    // read the 8-byte header of the file to make sure it's a png image
    unsigned char header[8];
    if (fread(header, 1, 8, file) < 1 || png_sig_cmp(header, 0, 8) != 0)
    {
        fprintf(stderr, "Error (%s) file is not a png\n", filename);
        fclose(file);
        return false;
    }
    
    // libpng uses 2 structures to keep track of the state of an open png file
    // these are the read struct and the info struct.  The next two function
    // calls dynamically allocate these structures
    libpng_read_struct = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (libpng_read_struct == NULL)
    {
        fprintf(stderr, "Error (%s) could not create png read struct\n", filename);
        fclose(file);
        return false;
    }
    
    libpng_info_struct = png_create_info_struct(libpng_read_struct);
    if (libpng_info_struct == NULL)
    {
        fprintf(stderr, "Error (%s) could not create png info struct\n", filename);
        png_destroy_read_struct(&libpng_read_struct, NULL, NULL);
        fclose(file);
        return false;
    }
    
    // set error handling.  This registers a way to get back here if an error
    // occurrs.  Had we not set this, libpng would abort the whole program
    // if an error occurred.
    if (setjmp(png_jmpbuf(libpng_read_struct)) != 0)
    {
        fprintf(stderr, "Error (%s) failed to setup longjump 1 for libpng\n", filename);
        fclose(file);
        png_destroy_info_struct(libpng_read_struct, &libpng_info_struct);
        png_destroy_read_struct(&libpng_read_struct, NULL, NULL);
        return false;
    }
    
    // initialize png for read operations
    png_init_io(libpng_read_struct, file);
    
    // let libpng know that we already read the first 8 bytes (for the header)
    png_set_sig_bytes(libpng_read_struct, 8);
    
    // initialize info struct
    png_read_info(libpng_read_struct, libpng_info_struct);
    
    /* End of yucky libpng initialization */
    
    // set the dimensions of the grid
    *width = png_get_image_width(libpng_read_struct, libpng_info_struct);
    *height = png_get_image_height(libpng_read_struct, libpng_info_struct);
    if (*width < 1 || *height < 1)
    {
        fprintf
        (
            stderr, "Error (%s) invalid dimensions width = %d, height = %d\n",
            filename,
            *width,
            *height
        );
        return 0;
    }
    
    // if rows are padded, each row may be larger than normal
    // calculate the memory footprint of each row
    unsigned row_bytes = (*width - 1) / 8 + 1;
    //unsigned padded_row_bytes = row_bytes + pad_width - (row_bytes % pad_width);
    
    // round up to nearest multiple of pad_width
    unsigned padded_row_bytes = row_bytes;
    unsigned mod = row_bytes % pad_width;
    if (mod != 0)
        padded_row_bytes += pad_width - mod;
    
    // allocate memory if the user did not pass a pre-allocated block
    if (*cell_grid == NULL)
        *cell_grid = new unsigned char[padded_row_bytes * *height];
    
    /* Actually read the pixel data from the png image */
    
    // pixels are read from the png file row by row and need temporary storage
    unsigned char* row_buff =
        new unsigned char[png_get_rowbytes(libpng_read_struct, libpng_info_struct)];
    
    switch (png_get_color_type(libpng_read_struct, libpng_info_struct))
    {
        /* 256-bit color lookup table */
        case PNG_COLOR_TYPE_PALETTE:
        {
            // search the color palette for the color black
            unsigned black_ix = 256;
            int num_colors;
            png_color_struct* palette;
            png_get_PLTE(libpng_read_struct, libpng_info_struct, &palette, &num_colors);
            for (int ix = 0; ix < num_colors; ix++)
            {
                png_color_struct* cur_color = palette + ix;
                if (cur_color->red == 0 && cur_color->green == 0 && cur_color->blue == 0)
                {
                    // the color palette contains an entry for a black pixel, save it
                    black_ix = ix;
                    break;
                }
            }
            
            // if the palette did not contain the color black, then black_ix will be 256 and
            // the grid will be initialized to all live cells
            
            switch (png_get_bit_depth(libpng_read_struct, libpng_info_struct))
            {
                
                // 8 bits per pixel (256-color image)
                case 8:
                {
                    // unrolled loop that fetches 8 pixels at a time since
                    // the cell_grid output array packs 8 cells into a single array element
                    unsigned unroll_bound = *width & ~0x7; // the first input ix that won't be
                                                           // processed in the unrolled loop
                    for (unsigned iy = 0; iy < *height; iy++)
                    {
                        
                        // read a row of pixels using libpng
                        png_read_row(libpng_read_struct, row_buff, NULL);
                        
                        unsigned char* output_base = *cell_grid + iy * padded_row_bytes;
                        for (unsigned ix = 0; ix < unroll_bound; ix += 8)
                        {
                            output_base[ix/8] = 
                                (row_buff[ix] != black_ix)
                              | ((row_buff[ix+1] != black_ix) << 1)
                              | ((row_buff[ix+2] != black_ix) << 2)
                              | ((row_buff[ix+3] != black_ix) << 3)
                              | ((row_buff[ix+4] != black_ix) << 4)
                              | ((row_buff[ix+5] != black_ix) << 5)
                              | ((row_buff[ix+6] != black_ix) << 6)
                              | ((row_buff[ix+7] != black_ix) << 7);
                        }
                        
                        // non-unrolled portion that fills in partial
                        // bytes at the right end of a row
                        unsigned char accumulator = 0;
                        for (unsigned ix = unroll_bound; ix < *width; ix++)
                        {
                            accumulator |= (row_buff[ix] != black_ix) << (ix - unroll_bound);
                        }
                        output_base[unroll_bound/8] = accumulator;
                    }
                    break;
                }
                
                // 1 bit per pixel (monochrome image)
                case 1:
                {
                    for (unsigned iy = 0; iy < *height; iy++)
                    {
                        // read a row of pixels using libpng
                        // libpng's format exactly matches ours, copy directly w/o buffer :)
                        
                        png_read_row
                        (
                            libpng_read_struct,
                            row_buff,
                            NULL
                        );
                        
                        // twist bits around
                        unsigned char* output_base = *cell_grid + iy * padded_row_bytes;
                        for (unsigned ix = 0, bound = (*width - 1) / 8 + 1; ix < bound; ix++)
                        {
                            
                            unsigned char read_byte = row_buff[ix];
                            unsigned char write_byte =
                             
                                ((read_byte & 0x01) << 7)
                              | ((read_byte & 0x02) << 5)
                              | ((read_byte & 0x04) << 3)
                              | ((read_byte & 0x08) << 1)
                              | ((read_byte & 0x10) >> 1)
                              | ((read_byte & 0x20) >> 3)
                              | ((read_byte & 0x40) >> 5)
                              | ((read_byte & 0x80) >> 7);
                            
                            // sometimes 0 is not black in the palette
                            if (black_ix != 0)
                                output_base[ix] = ~write_byte;
                            else
                                output_base[ix] = write_byte;
                        }
                        
                    }
                    
                    break;
                }
                
                // unsupported bit depth
                default:
                {
                    fprintf(stderr, "Error (%s) Bit depth must be 1 or 8.", filename);
                    png_destroy_info_struct(libpng_read_struct, &libpng_info_struct);
                    png_destroy_read_struct(&libpng_read_struct, NULL, NULL);
                    fclose(file);
                    delete[] row_buff;
                    return 0;
                }
                
            }
            
            break;
            
        }    

        /* 32-bit per pixel RGB + alpha */
        case PNG_COLOR_TYPE_RGB_ALPHA:
        {
            
            // unroll the *cell_grid init loop x8 since each elemnt contains 8 cells
            unsigned unroll_bound = (*width * 4) & ~0x1F; // 8 cells consumes 32 bytes
            
            // this function is already freakin-huge
            for (unsigned iy = 0; iy < *height; iy++)
            {
                
                // read a row of pixels using libpng
                // pixel colors are stored in row_buff as {red0, green0, blue0, alpha0, red1...
                png_read_row(libpng_read_struct, row_buff, NULL);
                
                unsigned char* output_base = *cell_grid + iy * padded_row_bytes;
                for (unsigned ix = 0; ix < unroll_bound; ix += 32)
                {
                    // prepare for super complicated indexing...
                    // don't bother reading alpha channel (3 + 4n)
                    output_base[ix / 32] = 
                      (((row_buff[ix] != 0) | (row_buff[ix+1] != 0) | (row_buff[ix+2] != 0)))
                    | (((row_buff[ix+4] != 0) | (row_buff[ix+5] != 0) | (row_buff[ix+6] != 0)) << 1)
                    | (((row_buff[ix+8] != 0) | (row_buff[ix+9] != 0) | (row_buff[ix+10] != 0)) << 2)
                    | (((row_buff[ix+12] != 0) | (row_buff[ix+13] != 0) | (row_buff[ix+14] != 0)) << 3)
                    | (((row_buff[ix+16] != 0) | (row_buff[ix+17] != 0) | (row_buff[ix+18] != 0)) << 4)
                    | (((row_buff[ix+20] != 0) | (row_buff[ix+21] != 0) | (row_buff[ix+22] != 0)) << 5)
                    | (((row_buff[ix+24] != 0) | (row_buff[ix+25] != 0) | (row_buff[ix+26] != 0)) << 6)
                    | (((row_buff[ix+28] != 0) | (row_buff[ix+29] != 0) | (row_buff[ix+30] != 0)) << 7);
                }
                // non-unrolled portion that fills in partial bytes at the right end of a row
                unsigned char accumulator = 0;
                for (unsigned ix = unroll_bound, last = *width * 4; ix < last; ix += 4)
                {
                    accumulator |=
                        (row_buff[ix] | row_buff[ix+1] | row_buff[ix+2]) // pixel is white
                     << ((ix - unroll_bound) / 4); // shift to position within byte
                }
                if (unroll_bound < *width * 4)
                    output_base[unroll_bound / 32] = accumulator;
            }
            
            break;
            
        }
        
        default:
        {
            fprintf(stderr, "Error (%s) PNG must be palletized or 32-bit RGBA\n", filename);
            png_destroy_info_struct(libpng_read_struct, &libpng_info_struct);
            png_destroy_read_struct(&libpng_read_struct, NULL, NULL);
            fclose(file);
            delete[] row_buff;
            return 0;
        }
    }    
    
    // cleanup
    png_destroy_info_struct(libpng_read_struct, &libpng_info_struct);
    png_destroy_read_struct(&libpng_read_struct, NULL, NULL);
    fclose(file);
    delete[] row_buff;
    
    return padded_row_bytes * *height;
    
}

bool pngrw_write_file
(
    const char* filename,
    const unsigned char* cell_grid,
    unsigned width,
    unsigned height,
    unsigned pad_width
){
    
    /* Yucky libpng initialization */
    
    // open file for reading in binary format
    FILE* file = fopen(filename, "wb");
    if (file == NULL)
    {
        fprintf(stderr, "Error (%s) failed to open file for writing\n", filename);
        return false;
    }
    
    // set error handling.  This registers a way to get back here if an error
    // occurrs.  Had we not set this, libpng would abort the whole program
    // if an error occurred.
    png_structp write_struct = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (write_struct == NULL)
    {
        fprintf(stderr, "Error (%s) could not create png write struct\n", filename);
        fclose(file);
        return false;
    }
    
    png_infop info_struct = png_create_info_struct(write_struct);
    if (info_struct == NULL)
    {
        fprintf(stderr, "Error (%s) could not create png info struct\n", filename);
        png_destroy_write_struct(&write_struct, &info_struct);
        fclose(file);
        return false;
    }

    // set error handling.  This registers a way to get back here if an error
    // occurrs.  Had we not set this, libpng would abort the whole program
    // if an error occurred.
    if (setjmp(png_jmpbuf(write_struct)) != 0)
    {
        fprintf(stderr, "Error (%s) failed to setup longjump 2 for libpng\n", filename);
        fclose(file);
        png_destroy_info_struct(write_struct, &info_struct);
        png_destroy_write_struct(&write_struct, NULL);
        return false;
    }
    
    png_init_io(write_struct, file);
    
    // write file header
    png_set_IHDR
    (
        write_struct,
        info_struct,
        width,
        height,
        1, // Bit depth
        PNG_COLOR_TYPE_GRAY,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_BASE,
        PNG_FILTER_TYPE_BASE
    );
    
    // write file header
    png_write_info(write_struct, info_struct);
    if (setjmp(png_jmpbuf(write_struct)) != 0)
	{
        fprintf(stderr, "Error (%s) failed to setup longjump 3 for libpng\n", filename);
        png_destroy_info_struct(write_struct, &info_struct);
        png_destroy_write_struct(&write_struct, NULL);
		fclose(file);
		return false;
	}
    
    // write pixel data
    if (setjmp(png_jmpbuf(write_struct)) != 0)
	{
        fprintf(stderr, "Error (%s) failed to setup longjump 4 for libpng\n", filename);
        png_destroy_info_struct(write_struct, &info_struct);
		png_destroy_write_struct(&write_struct, NULL);
		fclose(file);
		return false;
	}
    
    unsigned row_bytes = (width - 1) / 8 + 1;
    
    // round up to nearest multiple of pad_width
    unsigned padded_row_bytes = row_bytes;
    unsigned mod = row_bytes % pad_width;
    if (mod != 0)
        padded_row_bytes += pad_width - mod;
    unsigned char* row_buff = new unsigned char[padded_row_bytes];
    
    for (unsigned iy = 0; iy < height; iy++)
    {
        
        const unsigned char* input_base = cell_grid + iy * padded_row_bytes;
        
        // twist bits around
        for (unsigned ix = 0; ix < row_bytes; ix++)
        {
            unsigned char cur_byte = input_base[ix];
            row_buff[ix] = ((cur_byte & 0x01) << 7)
              | ((cur_byte & 0x02) << 5)
              | ((cur_byte & 0x04) << 3)
              | ((cur_byte & 0x08) << 1)
              | ((cur_byte & 0x10) >> 1)
              | ((cur_byte & 0x20) >> 3)
              | ((cur_byte & 0x40) >> 5)
              | ((cur_byte & 0x80) >> 7);
        }
        
        png_write_row(write_struct, row_buff);
        
    }
    
    // cleanup
    delete[] row_buff;
    png_write_end(write_struct, NULL);
    png_destroy_info_struct(write_struct, &info_struct);
    png_destroy_write_struct(&write_struct, NULL);
    fclose(file);
    
    return true;
    
}


