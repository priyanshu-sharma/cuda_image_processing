#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "kernel.cu"
using namespace cv;
using namespace std;

// void verify(float* input_h, float* output_h, float size)
// {
//     float *test_output = (float *) malloc(sizeof(float) * size);
//     float c = 255/log10(256);
//     for(int i = 0; i < size; i++)
//     {
//         test_output[i] = c * log(1 + input[i])
//     }
//     int count = 0;
//     for(int i = 0; i < size; i++)
//     {
//         if(test_output[i] != output_h[i])
//         {
//             cout<<"Difference in value - "<<i<<" - "<<test_output[i]<<" - "<<output_h[i]<<endl;
//             count = count + 1;
//         }
//     }
//     free(test_output);
//     if (count == 0)
//     {
//         cout<<"All Test Passed Successfully"<<endl;
//     }
// }

int main(int argc, char* argv[])
{
    cudaError_t cuda_ret;
    float *input_h, *output_h;
    float *input_d, *output_d;

    Mat image = imread("demo.png", IMREAD_GRAYSCALE);
    if (!image.data) { 
        printf("No image data \n");  
    }
    // int *myData = image.data;
    unsigned char *myData = (unsigned char*)(image.data);
    int width = image.cols;
    int height = image.rows;
    int stride = image.step;
    int image_size = width * height;
    cout<<"Image Size="<<image_size<<endl;
    cout<<"Width="<<width<<endl;
    cout<<"Height="<<height<<endl;
    cout<<"Stride="<<stride<<endl;

    input_h = (float *) malloc(sizeof(float) * image_size);
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            unsigned char val = myData[ i * stride + j];
            input_h[i * stride + j] = int(val);
        }
    }
    output_h = (float *) malloc(sizeof(float) * image_size);

    cudaMalloc((void **) &input_d, sizeof(float) * image_size);
    cudaMemcpy(input_d, input_h, sizeof(float) * image_size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &output_d, sizeof(float) * image_size);
    cudaDeviceSynchronize();

    log_transformation(input_d, output_d, image_size);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");

    // Copy device variables from host ----------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    cudaMemcpy(output_h, output_d, sizeof(float) * image_size, cudaMemcpyDeviceToHost);
    // verify(input_h, output_h, image_size);
    Mat input_image(height, width, CV_8UC1);
    Mat output_image(height, width, CV_8UC1);
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            input_image.at<uchar>(Point(j, i)) = input_h[i * stride + j];
            output_image.at<uchar>(Point(j, i)) = output_h[i * stride + j];
        }
    }
    bool in_check = imwrite("input.jpeg", input_image);
    if (!in_check)
    {
        cout<<"Failed To save input"<<endl;
    }
    bool out_check = imwrite("output.jpeg", output_image);
    if (!out_check)
    {
        cout<<"Failed To save output"<<endl;
    }
    free(input_h);
    free(output_h);
    cudaFree(input_d);
    cudaFree(output_d);
    return 0;
}
