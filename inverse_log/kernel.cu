#include <stdio.h>
# define BLOCK_SIZE 512


__global__ void inverse_log_kernel(double* input, double* output, double size)
{	
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    double c = log10f(256)/255;
    if (i < size)
    {
        output[i] = powf(10, c * input[i]) - 1;
    }
	/*************************************************************************/
}


void inverse_log(double* input, double* output, double size) {

	  /*************************************************************************/
    //INSERT CODE HERE
    dim3 DimGrid((size - 1)/BLOCK_SIZE + 1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    inverse_log_kernel<<<DimGrid, DimBlock>>>(input, output, size);
	  /*************************************************************************/
}