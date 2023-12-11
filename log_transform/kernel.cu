#include <stdio.h>
# define BLOCK_SIZE 512


__global__ void log_transformation_kernel(double* input, double* output, double size)
{	
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    printf("Double");
    printf(unsigned(log10f(256)));
    double c = 255 / log10f(256);
    if (i < size)
    {
        output[i] = c * log10f(1 + input[i]);
    }
	/*************************************************************************/
}


void log_transformation(double* input, double* output, double size) {

	  /*************************************************************************/
    //INSERT CODE HERE
    dim3 DimGrid((size - 1)/BLOCK_SIZE + 1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    log_transformation_kernel<<<DimGrid, DimBlock>>>(input, output, size);
	  /*************************************************************************/
}