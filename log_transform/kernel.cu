#include <stdio.h>
#include <bits/stdc++.h>
# define BLOCK_SIZE 512


__global__ void log_transformation_kernel(float* input, float* output, float size)
{
	
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    float c = 255/log10(256);
    if (i < size)
    {
        output[i] = c * log10(1 + input[i]);
    }
	/*************************************************************************/
}


void log_transformation(float* input, float* output, float size) {

	  /*************************************************************************/
    //INSERT CODE HERE
    dim3 DimGrid((size - 1)/BLOCK_SIZE + 1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    log_transformation_kernel<<<DimGrid, DimBlock>>>(input, output, size);
	  /*************************************************************************/
}