/*  Project:                ECE 408 Final Project
 *  File Name:              cuda_test.cu
 *  Calls:                  None
 *  Called by:              None
 *  Associated Header:      None
 *  Date created:           Sat Nov 7 2015
 *  Engineers:              Conor Gardner
 *  Compiler:               nvcc
 *  Target OS:              Ubuntu Linux 14.04
 *  Target architecture:    CPU:    x86 (64 bit)
 *                          GPU:    GeForce GTX 970 (compute capability 5.2)
 *  Description:            Simple vector add program designed to verify that your cuda
 *                          toolchain and drivers are properly installed */

#include <cstdio>

// cuda kernel executed on the GPU
__global__ void vector_add(
    unsigned vector_size,
    const float* input_a_d,
    const float* input_b_d,
    float* output_d
){
    
    unsigned ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if (ix < vector_size)
        output_d[ix] = input_a_d[ix] + input_b_d[ix];
    
}

// CPU code which sends the GPU work and verifies the output
int main()
{
    
    // feel free to customize these parameters
    const unsigned vector_size = 1024;
    const unsigned threads_per_block = 256;
    const float tolerance = 0.01;           // maximum floating-point error when verifying (1%)
    
    // initialize memory on the host (just initilize with arbitrary data)
    float* input_a_h = new float[vector_size];
    float* input_b_h = new float[vector_size];
    float* output_h = new float[vector_size];
    for (unsigned ix = 0; ix < vector_size; ix++)
    {
        input_a_h[ix] = ix * (ix / 3.3f);
        input_b_h[ix] = 7.8f * ix + 8.1f;
    }
    
    // allocate memory on the device and perform host
    float* input_a_d;
    float* input_b_d;
    float* output_d;
    cudaMalloc(&input_a_d, vector_size * sizeof(float));
    cudaMalloc(&input_b_d, vector_size * sizeof(float));
    cudaMalloc(&output_d, vector_size * sizeof(float));
    
    // blocking memory copy host --> device
    cudaMemcpy(input_a_d, input_a_h, vector_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(input_b_d, input_b_h, vector_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_d, output_h, vector_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // invoke the GPU - it will perform the vector addition after the previous memcpy
    unsigned blocks_per_grid = (vector_size - 1) / threads_per_block + 1;
    vector_add<<<blocks_per_grid, threads_per_block>>>(vector_size, input_a_d, input_b_d, output_d);
    
    // blocking memory copy device --> host
    cudaMemcpy(input_a_h, input_a_d, vector_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(input_b_h, input_b_d, vector_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_h, output_d, vector_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // de-allocate device memory ASAP
    cudaFree(input_a_d);
    cudaFree(input_b_d);
    cudaFree(output_d);
    
    // verify the data received from the device
    float max_percent_err = 0.0f;
    for (unsigned ix = 0; ix < vector_size; ix++)
    {
        
        float host_result = input_a_h[ix] + input_b_h[ix];
        float device_result = output_h[ix];
        
        float percent_err = (host_result - device_result) / host_result;
        if (percent_err < 0.0f)
            percent_err = -percent_err;
        
        if (percent_err > max_percent_err)
            max_percent_err = percent_err;
        
        // print error message and abort on the first incorrect result
        if (percent_err > tolerance)
        {
            printf
            (
                "FAILED to verify element %u: %f + %f = %f.  GPU returned %f\n",
                ix,
                input_a_h[ix],
                input_b_h[ix],
                host_result,
                device_result
            );
            return -1;
        }
        
    }
    printf("SUCCESS with max error = %3.2f%%\n", max_percent_err * 100.0);
    
    // de-allocate host memory
    delete[] input_a_h;
    delete[] input_b_h;
    delete[] output_h;
    
    return 0;
    
}
