#include <stdio.h>
# define BLOCK_SIZE 512


__global__ void image_thresholding_kernel(double* input, double* output, double size)
{	
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size)
    {
        if (input[i] < 128)
        {
            output[i] = 0;
        }
        else
        {
            output[i] = 255;
        }
    }
	/*************************************************************************/
}


void image_thresholding(double* input, double* output, double size) {

	  /*************************************************************************/
    //INSERT CODE HERE
    dim3 DimGrid((size - 1)/BLOCK_SIZE + 1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    image_thresholding_kernel<<<DimGrid, DimBlock>>>(input, output, size);
	  /*************************************************************************/
}