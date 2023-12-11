#include <stdio.h>
#include <math.h>
# define BLOCK_SIZE 512


__global__ void log_transformation_kernel(float* input, float* output, float size)
{
	
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    __constant__ float c = 255 / (log(256)/log(10));
    if (i < size)
    {
        output[i] = c * log(1 + input[i]);
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