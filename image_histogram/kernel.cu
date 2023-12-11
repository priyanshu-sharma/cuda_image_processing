#include <stdio.h>
# define BLOCK_SIZE 512
# define MAX_NUMBER_OF_BLOCK 16
# define COLOR_LEVEL 255

__global__ void image_histogram_kernel(double* input, int size, double* histogram, double *output, double *cdf, double *final_output, int total_bins)
{
	
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    __shared__ int local_ihisto[256];
    int i, stride;
    for ( i = threadIdx.x ; i < total_bins ; i += BLOCK_SIZE )
    {
        local_ihisto[i] = 0;
    }
    __syncthreads();

    i = threadIdx.x + blockIdx.x * blockDim.x;
    stride = blockDim.x * gridDim.x;
    while ( i < size )
    {
        int value = input[i];
        atomicAdd(&(local_ihisto[value]), 1);
        i += stride;
    }

    __syncthreads();
    for ( i = threadIdx.x ; i < total_bins ; i += BLOCK_SIZE )
    {
        atomicAdd(&(histogram[i]), local_ihisto[i]);
    }
    __syncthreads();
    for ( i = threadIdx.x ; i < total_bins ; i += BLOCK_SIZE )
    {
        output[i] = histogram[i]/size;
    }
    __syncthreads();
    double sum = 0;
    i = 0;
    while(i < threadIdx.x + 1)
    {
        sum = sum + output[i];
        i = i + 1;
    }
    cdf[ threadIdx.x ] = floorf(COLOR_LEVEL * sum);
    __syncthreads();
    i = threadIdx.x + blockIdx.x * blockDim.x;
    stride = blockDim.x * gridDim.x;
    while ( i < size )
    {
        int value = input[i];
        final_output[i] = cdf[value];
        i += stride;
    }
    /*************************************************************************/
}


void image_histogram(double* input, int size, double* histogram, double *output, double *cdf, double *final_output, int total_bins) {

	  /*************************************************************************/
    //INSERT CODE HERE
    int totalBlocks = (size - 1)/BLOCK_SIZE + 1;
    if ( totalBlocks > MAX_NUMBER_OF_BLOCK )
    {
        totalBlocks = MAX_NUMBER_OF_BLOCK;
    }
    dim3 DimGrid(totalBlocks, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    image_histogram_kernel<<<DimGrid, DimBlock>>>(input, size, histogram, output, cdf, final_output, total_bins);
	  /*************************************************************************/
}