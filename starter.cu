#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

# define BLOCK_SIZE 512


__global__ void scaling_kernel(unsigned int* input, unsigned int size)
{
	
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size)
    {
        input[i] = 2 * input[i];
    }
	/*************************************************************************/
}


void scaling(unsigned int* input, unsigned int size) {

	  /*************************************************************************/
    //INSERT CODE HERE
    dim3 DimGrid((size - 1)/BLOCK_SIZE + 1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    scaling_kernel<<<DimGrid, DimBlock>>>(input, size);
	  /*************************************************************************/
}


int main(int argc, char* argv[])
{
    cudaError_t cuda_ret;
    Mat image = imread("demo.png", IMREAD_GRAYSCALE);
    if (!image.data) { 
        printf("No image data \n");  
    }
    uint8_t *myData = image.data;
    int width = image.cols;
    int height = image.rows;
    int _stride = image.step;
    unsigned int size = width * height * sizeof(unsigned int);
    cout<<"Size="<<size<<endl;
    cout<<"Width="<<unsigned(width)<<endl;
    cout<<"Height="<<unsigned(height)<<endl;
    cout<<"Stride="<<unsigned(_stride)<<endl;
    unsigned int *image_vector = (unsigned int *) malloc(size);
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            uint8_t val = myData[ i * _stride + j];
            image_vector[i * _stride + j] = unsigned(val);
        }
    }
    unsigned int *image_vector_d;
    cudaMalloc((void **) &image_vector_d, size);
    unsigned int *o_image_vector = (unsigned int *) malloc(size);
    cudaMemcpy(image_vector_d, image_vector, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    scaling(image_vector_d, size);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");

    // Copy device variables from host ----------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    cudaMemcpy(o_image_vector, image_vector_d, size, cudaMemcpyDeviceToHost);
    cout<<o_image_vector[262143]<<" - "<<image_vector[262143]<<endl;
    free(image_vector);
    cudaFree(o_image_vector);
    return 0;
}