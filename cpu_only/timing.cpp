/*  Project:                ECE 408 Final Project
 *  File Name:              timing.cpp
 *  Calls:                  none
 *  Called by:              main.cpp
 *  Associated Header:      ece408_final_cpu.h
 *  Date created:           Sat Nov 7 2015
 *  Engineers:              Conor Gardner
 *  Compiler:               g++
 *  Target OS:              Ubuntu Linux 14.04
 *  Target architecture:    x86 (64 bit) */

#include <sys/time.h>

// returns the number of seconds since some arbitrary starting point with microsecond precision
double get_timestamp()
{
    struct timeval stamp;
    gettimeofday(&stamp, 0);
    return (double)(stamp.tv_sec) + (double)(stamp.tv_usec) / 1000000.0;
}

