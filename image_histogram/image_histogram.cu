#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "kernel.cu"
using namespace cv;
using namespace std;

void verify(unsigned int* input_h, unsigned size, unsigned int* histogram_h, unsigned int total_bins)
{
    unsigned int *test_histogram = (unsigned int *) malloc(sizeof(unsigned int) * total_bins);
    for(int i = 0; i < total_bins; i++)
    {
        test_histogram[i] = 0;
    }
    for(int i = 0; i < size; i++)
    {
        test_histogram[input_h[i]] = test_histogram[input_h[i]] + 1;
    }
    for(int i = 0; i < total_bins; i++)
    {
        if(test_histogram[i] != histogram_h[i])
        {
            cout<<"Difference in value - "<<i<<" - "<<test_histogram[i]<<" - "<<histogram_h[i]<<endl;
        }
    }
    free(test_histogram);
}

int main(int argc, char* argv[])
{
    cudaError_t cuda_ret;
    unsigned int *input_h, *histogram_h;
    unsigned int *input_d, *histogram_d;
    unsigned int total_bins = 256;

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
    histogram_h = (unsigned int *) malloc(sizeof(unsigned int) * total_bins);

    cudaMalloc((void **) &input_d, sizeof(unsigned int) * image_size);
    cudaMemcpy(input_d, input_h, sizeof(unsigned int) * image_size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &histogram_d, sizeof(unsigned int) * total_bins);
    cudaMemset(histogram_d, 0, total_bins * sizeof(unsigned int));
    cudaDeviceSynchronize();

    image_histogram(input_d, image_size, histogram_d, total_bins);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");

    // Copy device variables from host ----------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    cudaMemcpy(histogram_h, histogram_d, sizeof(unsigned int) * total_bins, cudaMemcpyDeviceToHost);
    // for(int i = 0; i < total_bins; i++)
    // {
    //     cout<<i<<" - "<<histogram_h[i]<<endl;
    // }
    verify(input_h, image_size, histogram_h, total_bins);
    free(input_h);
    free(histogram_h);
    cudaFree(input_d);
    cudaFree(histogram_d);
    return 0;
}
