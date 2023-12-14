#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "kernel.cu"

using namespace cv;
using namespace std;

// Forward declaration of the Gaussian blur function
void gaussianBlur(unsigned char *input, unsigned char *output, int width, int height, float sigma);

int main(int argc, char* argv[])
{
    const char* inputFile = "demo.png";  // Input file name
    const char* outputFile = "output.png";  // Output file name
    float sigma = 1.0; // Example sigma value for Gaussian kernel

    // Read the image
    Mat image = imread(inputFile, IMREAD_GRAYSCALE);
    if (!image.data) { 
        printf("No image data \n");
        return -1;
    }
    uint8_t *myData = image.data;
    int width = image.cols;
    int height = image.rows;
    int stride = image.step;
    unsigned int image_size = width * height;

    // Allocate memory for input and output images on host
    unsigned char *input_h, *output_h;
    input_h = (unsigned char *)malloc(image_size * sizeof(unsigned char));
    output_h = (unsigned char *)malloc(image_size * sizeof(unsigned char));

    // Copy image data from OpenCV input image to the input array
    for (int i = 0; i < height; i++) {
        memcpy(&input_h[i * width], &myData[i * stride], width);
    }

    // Start CPU timer
    clock_t start_cpu = clock();

    // Start CUDA timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Call the Gaussian blur function
    gaussianBlur(input_h, output_h, width, height, sigma);

    // Stop CUDA timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Stop CPU timer
    clock_t stop_cpu = clock();

    // Calculate and print the elapsed CPU time
    double duration_cpu = (double)(stop_cpu - start_cpu) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds
    printf("CPU Time: %.3f ms\n", duration_cpu);
    printf("CUDA Time: %.3f ms\n", milliseconds);

    // Create an empty Mat to store the result
    Mat result(height, width, CV_8U, output_h);
    // Save the result to a file
    imwrite(outputFile, result);

    // Free host memory
    free(input_h);
    free(output_h);

    printf("Gaussian blur completed successfully.\n");

    return 0;
}