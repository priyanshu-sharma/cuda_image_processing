#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "kernel.cu"
using namespace cv;
using namespace std;

void verify(unsigned int* input_h, unsigned int* output_h, unsigned int size)
{
    unsigned int *test_output = (unsigned int *) malloc(sizeof(unsigned int) * size);
    for(int i = 0; i < size; i++)
    {
        test_output[i] = 255 - input_h[i];
    }
    unsigned int count = 0;
    for(int i = 0; i < size; i++)
    {
        if(test_output[i] != output_h[i])
        {
            cout<<"Difference in value - "<<i<<" - "<<test_output[i]<<" - "<<output_h[i]<<endl;
            count = count + 1;
        }
    }
    free(test_output);
    if (count == 0)
    {
        cout<<"All Test Passed Successfully"<<endl;
    }
}

int main(int argc, char* argv[])
{
    cudaError_t cuda_ret;
    unsigned int *input_h, *output_h;
    unsigned int *input_d, *output_d;

    Mat image = imread("demo.png", IMREAD_GRAYSCALE);
    if (!image.data) { 
        printf("No image data \n");  
    }
    uint8_t *myData = image.data;
    int width = image.cols;
    int height = image.rows;
    int _stride = image.step;
    unsigned int image_size = width * height;
    cout<<"Image Size="<<image_size<<endl;
    cout<<"Width="<<unsigned(width)<<endl;
    cout<<"Height="<<unsigned(height)<<endl;
    cout<<"Stride="<<unsigned(_stride)<<endl;

    input_h = (unsigned int *) malloc(sizeof(unsigned int) * image_size);
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            uint8_t val = myData[ i * _stride + j];
            input_h[i * _stride + j] = unsigned(val);
        }
    }
    output_h = (unsigned int *) malloc(sizeof(unsigned int) * image_size);

    cudaMalloc((void **) &input_d, sizeof(unsigned int) * image_size);
    cudaMemcpy(input_d, input_h, sizeof(unsigned int) * image_size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &output_d, sizeof(unsigned int) * image_size);
    cudaDeviceSynchronize();

    inverted_image(input_d, output_d, image_size);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");

    // Copy device variables from host ----------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    cudaMemcpy(output_h, output_d, sizeof(unsigned int) * image_size, cudaMemcpyDeviceToHost);
    verify(input_h, output_h, image_size);
    free(input_h);
    free(output_h);
    cudaFree(input_d);
    cudaFree(output_d);
    return 0;
}
