#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "kernel.cu"


int main()
{
    // Load image using OpenCV
    cv::Mat image = cv::imread("demo.png", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "Could not open or find the image." << std::endl;
        return -1;
    }

    // Get image parameters
    int width = image.cols;
    int height = image.rows;

    // Allocate memory for input and output images
    unsigned char* input_h = image.data;
    unsigned char* output_h = (unsigned char*)malloc(sizeof(unsigned char) * width * height);

    unsigned char *input_d, *output_d;
    cudaMalloc((void**)&input_d, sizeof(unsigned char) * width * height);
    cudaMalloc((void**)&output_d, sizeof(unsigned char) * width * height);

    // Copy input image data to device
    cudaMemcpy(input_d, input_h, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);

    // Perform image smoothing
    imageSmoothing(input_d, output_d, width, height);

    // Copy result back to host
    cudaMemcpy(output_h, output_d, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

    // Save the input and output images
    cv::imwrite("input_image.jpg", image);
    cv::imwrite("output_smoothed_image.jpg", cv::Mat(height, width, CV_8UC1, output_h));

    // Free allocated memory
    free(output_h);
    cudaFree(input_d);
    cudaFree(output_d);

    return 0;
}