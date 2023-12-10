#include <stdio.h>
# define BLOCK_SIZE 512


__global__ void inverted_image_kernel(unsigned int* input, unsigned int* output, unsigned int size)
{
	
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size)
    {
        output[i] = 255 - input[i];
    }
	/*************************************************************************/
}


void inverted_image(unsigned int* input, unsigned int* output, unsigned int size) {

	  /*************************************************************************/
    //INSERT CODE HERE
    dim3 DimGrid((size - 1)/BLOCK_SIZE + 1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    inverted_image_kernel<<<DimGrid, DimBlock>>>(input, output, size);
	  /*************************************************************************/
}