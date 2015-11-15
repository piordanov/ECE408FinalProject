/*  Project:                ECE 408 Final Project
 *  File Name:              utility.cpp
 *  Calls:                  none
 *  Called by:              main.cpp
 *  Associated Header:      ece408_final_gpu.h
 *  Date created:           Sat Nov 7 2015
 *  Engineers:              Peter Iordanov, Laura Galbraith, Conor Gardner
 *  Compiler:               nvcc
 *  Target OS:              Ubuntu Linux 14.04
 *  Target architecture:    CPU: x86 (64 bit)       GPU: Nvida GTX 970*/

#include <cstdio>
#include <cstring>
#include <sys/time.h>
#include "ece408_final_gpu.h"

using namespace std;

// returns the number of seconds since some arbitrary starting point with microsecond precision
double get_timestamp()
{
    struct timeval stamp;
    gettimeofday(&stamp, 0);
    return (double)(stamp.tv_sec) + (double)(stamp.tv_usec) / 1000000.0;
}

    
// constructor for output_filename_t class
output_filename_t::output_filename_t(const char* output_pattern, unsigned iterations)
{
    
    // parameter check.  Can remove if you're sure the inputs will always be valid
    if (iterations == 0 || output_pattern == NULL)
    {
        fprintf(stderr, "Warning! Created empty output_filename_t object\n");
        str = NULL;
        return;
    }
    
    // count the minimum number of decimal digits required to print "iterations"
    // digits = 1   if 1 <= iterations < 10
    // digits = 2   if 10 <= iterations < 100
    unsigned digits = 0;
    do
    {
        iterations /= 10;
        digits++;
    } while (iterations != 0);
    
    // calculate the length of the final filename string
    unsigned prefix_len = strlen(output_pattern);
    unsigned str_len =
        prefix_len         // "output_pattern
      + digits             //  0000
      + 6;                 //  _ .png \0"
    
    str = new char[str_len];
    
    // concatenate pieces to form output string
    memcpy(str, output_pattern, prefix_len * sizeof(char));
    str[prefix_len] = '_';
    memset(str + prefix_len + 1, '0', digits * sizeof(char));
    str[prefix_len + 1 + digits] = '.';
    str[prefix_len + 2 + digits] = 'p';
    str[prefix_len + 3 + digits] = 'n';
    str[prefix_len + 4 + digits] = 'g';
    str[prefix_len + 5 + digits] = '\0';
    
    digit_end = prefix_len + digits;
    
}

// copy constructor for output_filename_t class
output_filename_t::output_filename_t(const output_filename_t& rhs)
{
    
    // create an invalid object if the input is invalid
    if (rhs.str == NULL)
    {
        fprintf(stderr, "Warning! Copied empty output_filename_t object!\n");
        str = NULL;
        return;
    }
    
    // typical case - input is valid
    digit_end = rhs.digit_end;
    len = rhs.len;
    str = new char[len];
    memcpy(str, rhs.str, len);
    
}

// assignment operator for output_filename_t class
output_filename_t& output_filename_t::operator=(const output_filename_t& rhs)
{
    
    delete[] str;
    
    // create an invalid object if the input is invalid
    if (rhs.str == NULL)
    {
        fprintf(stderr, "Warning! Assigned empty output_filename_t object!\n");
        str = NULL;
        return *this;
    }
    
    // typical case - input is valid
    digit_end = rhs.digit_end;
    len = rhs.len;
    str = new char[len];
    memcpy(str, rhs.str, len);
    
    return *this;
    
}

// destructor for output_filename_t class
output_filename_t::~output_filename_t()
{
    
    if (str == NULL)
        fprintf(stderr, "Warning! Deleted empty output_filename_t object!\n");
    else
        delete[] str;
}

// output_filename_t class
// example: advances "output_042.png" to "output_043.png"
// returns true on success, false if an error occured
bool output_filename_t::next_filename()
{
    
    // abort for uninitialized objects
    if (str == NULL)
    {
        fprintf(stderr, "Warning! Incremented empty output_filename_t object!\n");
        return false;
    }
    
    // ASCII "ripple adder".
    // example: if str starts off as "output_4299.png" it will become "output_4300.png"
    // after 3 iterations
    unsigned ix = digit_end;
    while (ix > 0)
    {
        // sorry for the weird control flow ;)
        char c = str[ix];
        if (c >= '0' && c <= '8')
        {
            str[ix] = c + 1;
            return true;
        } else if (c == '9') {
            str[ix] = '0';
            ix--;
            continue;
        } else {
            fprintf(stderr, "Warning! output_filename_t incremented past last filename\n");
            return false;
        }
    }
    
    fprintf(stderr, "Warning! output_filename_t incremented past first character!\n");
    return false;
    
}

