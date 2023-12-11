#include <opencv2/opencv.hpp>
#include <iostream>
#include "kernel.cu"

#define BLOCK_SIZE 16

int main()
{
    // Load the image using OpenCV
    cv::Mat image = cv::imread("demo.png", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "Could not open or find the image." << std::endl;
        return -1;
    }

    // Get image properties
    int width = image.cols;
    int height = image.rows;

    // Allocate memory for input and output images
    unsigned char* input_h = image.data;
    unsigned char* output_h = (unsigned char*)malloc(sizeof(unsigned char) * width * height);
    unsigned char* input_d, *output_d;

    // Allocate memory on the GPU
    cudaMalloc((void**)&input_d, sizeof(unsigned char) * width * height);
    cudaMalloc((void**)&output_d, sizeof(unsigned char) * width * height);

    // Copy input image data to the GPU
    cudaMemcpy(input_d, input_h, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);

    // Apply image blur using CUDA
    imageBlur(input_d, output_d, width, height);

    // Copy the result back to the host
    cudaMemcpy(output_h, output_d, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

    // Create a Mat object for the output image
    cv::Mat output_image(height, width, CV_8UC1, output_h);

    // Save the output image
    cv::imwrite("output_blurred.jpg", output_image);

    // Free allocated memory
    free(output_h);
    cudaFree(input_d);
    cudaFree(output_d);

    return 0;
}
