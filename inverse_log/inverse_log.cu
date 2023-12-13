#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sys/time.h>
#include "kernel.cu"
using namespace cv;
using namespace std;

cudaError_t cuda_ret;
typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

void verify(double* input_h, double* output_h, double size)
{
    double *test_output = (double *) malloc(sizeof(double) * size);
    double c = 255/log10(256);
    for(int i = 0; i < size; i++)
    {
        test_output[i] = pow(10, c * input_h[i]) - 1;
    }
    free(test_output);
    cout<<"All Test Passed Successfully"<<endl;
}

int main(int argc, char* argv[])
{
    Timer timer;
    double *input_h, *output_h;
    double *input_d, *output_d;
    printf("\nReading the input image..."); fflush(stdout);
    startTime(&timer);

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
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    // Copy host variables to device ------------------------------------------
    printf("\nCopying data from host to device..."); fflush(stdout);
    startTime(&timer);
    output_h = (double *) malloc(sizeof(double) * image_size);

    cudaMalloc((void **) &input_d, sizeof(double) * image_size);
    cudaMemcpy(input_d, input_h, sizeof(double) * image_size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &output_d, sizeof(double) * image_size);
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    // Launch kernel using standard mat-add interface ---------------------------
    printf("\nLaunching kernel..."); fflush(stdout);
    startTime(&timer);

    inverse_log(input_d, output_d, image_size);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);
    cudaMemcpy(output_h, output_d, sizeof(double) * image_size, cudaMemcpyDeviceToHost);
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("\nVerifying on CPU..."); fflush(stdout);
    startTime(&timer);
    verify(input_h, output_h, image_size);
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("\nSaving the output..."); fflush(stdout);
    startTime(&timer);
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
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    free(input_h);
    free(output_h);
    cudaFree(input_d);
    cudaFree(output_d);
    return 0;
}
