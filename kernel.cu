#include <stdio.h>
# define BLOCK_SIZE 512


__global__ void scaling_kernel(unsigned int* input, unsigned int* output, unsigned int size)
{
	
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size)
    {
        output[i] = 2 * input[i];
    }
	/*************************************************************************/
}


void scaling(unsigned int* input, unsigned int* output, unsigned int size) {

	  /*************************************************************************/
    //INSERT CODE HERE
    dim3 DimGrid((size - 1)/BLOCK_SIZE + 1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    scaling_kernel<<<DimGrid, DimBlock>>>(input, output, size);
	  /*************************************************************************/
}