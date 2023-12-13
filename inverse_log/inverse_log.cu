#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>
#include "kernel.cu"
using namespace cv;
using namespace std;

// void verify(double* input_h, double* output_h, double size)
// {
//     double *test_output = (double *) malloc(sizeof(double) * size);
//     double c = 255/log10(256);
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
    double *input_h, *output_h;
    double *input_d, *output_d;
    printf("\nReading the input image..."); fflush(stdout);

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

    input_h = (double *) malloc(sizeof(double) * image_size);
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            unsigned char val = myData[ i * stride + j];
            input_h[i * stride + j] = int(val);
        }
    }
    // Copy host variables to device ------------------------------------------
    printf("\nCopying data from host to device..."); fflush(stdout);
    output_h = (double *) malloc(sizeof(double) * image_size);

    cudaMalloc((void **) &input_d, sizeof(double) * image_size);
    cudaMemcpy(input_d, input_h, sizeof(double) * image_size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &output_d, sizeof(double) * image_size);
    cudaDeviceSynchronize();
    // Launch kernel using standard mat-add interface ---------------------------
    printf("\nLaunching kernel..."); fflush(stdout);
    time_t start = time(0);
    cout<<"Start Time - "<<start<<endl;

    inverse_log(input_d, output_d, image_size);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");
    time_t end = time(0);
    cout<<"End Time - "<<end<<endl;
    cout<<"\nTotal Time - "<<end-start<<endl;

    // Copy device variables from host ----------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    cudaMemcpy(output_h, output_d, sizeof(double) * image_size, cudaMemcpyDeviceToHost);
    // verify(input_h, output_h, image_size);
    printf("\nSaving the output..."); fflush(stdout);
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
