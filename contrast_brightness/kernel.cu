#include <stdio.h>
# define BLOCK_SIZE 512


__global__ void contrast_brightness_kernel(double* input, double* output, double size)
{
	
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size)
    {
        output[i] = 2 * input[i] + 4;
    }
	/*************************************************************************/
}


void contrast_brightness(double* input, double* output, double size) {

	  /*************************************************************************/
    //INSERT CODE HERE
    dim3 DimGrid((size - 1)/BLOCK_SIZE + 1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    contrast_brightness_kernel<<<DimGrid, DimBlock>>>(input, output, size);
	  /*************************************************************************/
}