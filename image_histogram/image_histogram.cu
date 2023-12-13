#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>
#include "kernel.cu"
using namespace cv;
using namespace std;

// void verify(unsigned int* input_h, unsigned size, unsigned int* histogram_h, unsigned int total_bins)
// {
//     unsigned int *test_histogram = (unsigned int *) malloc(sizeof(unsigned int) * total_bins);
//     for(int i = 0; i < total_bins; i++)
//     {
//         test_histogram[i] = 0;
//     }
//     for(int i = 0; i < size; i++)
//     {
//         test_histogram[input_h[i]] = test_histogram[input_h[i]] + 1;
//     }
//     unsigned int count = 0;
//     for(int i = 0; i < total_bins; i++)
//     {
//         if(test_histogram[i] != histogram_h[i])
//         {
//             cout<<"Difference in value - "<<i<<" - "<<test_histogram[i]<<" - "<<histogram_h[i]<<endl;
//             count = count + 1;
//         }
//     }
//     free(test_histogram);
//     if (count == 0)
//     {
//         cout<<"All Test Passed Successfully"<<endl;
//     }
// }

int main(int argc, char* argv[])
{
    cudaError_t cuda_ret;
    double *input_h, *histogram_h, *output_h, *cdf_h, *final_output_h, *ff;
    double *input_d, *histogram_d, *output_d, *cdf_d, *final_output_d;
    int total_bins = 256;
    printf("\nReading the input image..."); fflush(stdout);

    Mat image = imread("demo.png", IMREAD_GRAYSCALE);
    if (!image.data) { 
        printf("No image data \n");  
    }
    unsigned char *myData = (unsigned char*)(image.data);
    int width = image.cols;
    int height = image.rows;
    int stride = image.step;
    int image_size = width * height;
    cout<<"Image Size="<<image_size<<endl;
    cout<<"Width="<<unsigned(width)<<endl;
    cout<<"Height="<<unsigned(height)<<endl;
    cout<<"Stride="<<unsigned(stride)<<endl;

    input_h = (double*) malloc(sizeof(double) * image_size);
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
    histogram_h = (double *) malloc(sizeof(double) * total_bins);
    output_h = (double *) malloc(sizeof(double) * total_bins);
    cdf_h = (double *) malloc(sizeof(double) * total_bins);
    final_output_h = (double *) malloc(sizeof(double) * image_size);
    ff = (double *) malloc(sizeof(double) * image_size);

    cudaMalloc((void **) &input_d, sizeof(double) * image_size);
    cudaMemcpy(input_d, input_h, sizeof(double) * image_size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &histogram_d, sizeof(double) * total_bins);
    cudaMemset(histogram_d, 0, total_bins * sizeof(double));
    cudaMalloc((void **) &output_d, sizeof(double) * total_bins);
    cudaMalloc((void **) &cdf_d, sizeof(double) * total_bins);
    cudaMalloc((void **) &final_output_d, sizeof(double) * image_size);
    cudaDeviceSynchronize();
    // Launch kernel using standard mat-add interface ---------------------------
    printf("\nLaunching kernel..."); fflush(stdout);
    time_t start = time(0);
    cout<<"Start Time - "<<start<<endl;

    image_histogram(input_d, image_size, histogram_d, output_d, cdf_d, final_output_d, total_bins);
    time_t end = time(0);
    cout<<"End Time - "<<end<<endl;
    cout<<"\nTotal Time - "<<end-start<<endl;
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel");

    // Copy device variables from host ----------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    cudaMemcpy(histogram_h, histogram_d, sizeof(double) * total_bins, cudaMemcpyDeviceToHost);
    cudaMemcpy(output_h, output_d, sizeof(double) * total_bins, cudaMemcpyDeviceToHost);
    cudaMemcpy(cdf_h, cdf_d, sizeof(double) * total_bins, cudaMemcpyDeviceToHost);
    cudaMemcpy(final_output_h, final_output_d, sizeof(double) * image_size, cudaMemcpyDeviceToHost);
    cout<<"\nImage Histogram Distribution\n"<<endl;
    for(int i = 0; i < total_bins; i++)
    {
        cout<<i<<" - "<<histogram_h[i]<<" - "<<output_h[i]<<" - "<<cdf_h[i]<<endl;
    }
    for(int i = 0; i < image_size; i++)
    {
        int finalv = input_h[i];
        ff[i] = cdf_h[finalv];

    }
    printf("\nSaving the output..."); fflush(stdout);
    Mat input_image(height, width, CV_8UC1);
    Mat output_image(height, width, CV_8UC1);
    Mat ff_im(height, width, CV_8UC1);
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            input_image.at<uchar>(Point(j, i)) = input_h[i * stride + j];
            output_image.at<uchar>(Point(j, i)) = final_output_h[i * stride + j];
            ff_im.at<uchar>(Point(j, i)) = ff[i * stride + j];
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
    bool ff_check = imwrite("ff.jpeg", ff_im);
    if (!ff_check)
    {
        cout<<"Failed To save ff"<<endl;
    }
    // verify(input_h, image_size, histogram_h, total_bins);
    free(input_h);
    free(histogram_h);
    free(output_h);
    free(cdf_h);
    free(final_output_h);
    free(ff);
    cudaFree(input_d);
    cudaFree(histogram_d);
    cudaFree(output_d);
    cudaFree(cdf_d);
    cudaFree(final_output_d);
    return 0;
}
