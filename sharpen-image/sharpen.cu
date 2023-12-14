#include <iostream>
#include <opencv2/opencv.hpp>
#include "kernel.cu" 
#include <ctime>

using namespace cv;
using namespace std;

// Forward declaration of the sharpening function
void sharpen_image(unsigned char *input, unsigned char *output, int width, int height, int channels);

int main(int argc, char* argv[])
{
    const char* inputFile = "demo.png";
    const char* outputFile = "sharpen_image.png";

    // Read the image in color
    Mat image = imread(inputFile, IMREAD_COLOR);
    if (image.empty()) { 
        cerr << "Could not open or find the image\n";
        return -1;
    }

    // Ensure the image is in the expected format
    if(image.type() != CV_8UC3) {
        cerr << "Image format must be 8-bit, 3 channels.\n";
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();
    unsigned int image_size = width * height * channels;

    // Allocate memory for input and output images on host
    unsigned char *input_h, *output_h;
    input_h = (unsigned char *)malloc(image_size * sizeof(unsigned char));
    output_h = (unsigned char *)malloc(image_size * sizeof(unsigned char));

    // Copy image data from OpenCV input image to the input array
    memcpy(input_h, image.data, image_size);

    // Start CPU timer
    clock_t start_cpu = clock();

    // Start CUDA timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Call the sharpen function
    sharpen_image(input_h, output_h, width, height, channels);

    // Stop CUDA timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Stop CPU timer
    clock_t stop_cpu = clock();

    // Calculate and print the elapsed CPU time
    double duration_cpu = (double)(stop_cpu - start_cpu) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds
    cout << "CPU Time: " << duration_cpu << " ms\n";
    cout << "CUDA Time: " << milliseconds << " ms\n";

    // Create an empty Mat to store the result
    Mat result(height, width, CV_8UC3, output_h);
    // Save the result to a file
    imwrite(outputFile, result);

    // Free host memory
    free(input_h);
    free(output_h);

    cout << "Image sharpening completed successfully." << endl;

    return 0;
}