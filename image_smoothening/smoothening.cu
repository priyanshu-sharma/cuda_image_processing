#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include "kernel.cu"


// CPU implementation of image smoothing
void imageSmoothingCPU(const unsigned char* input, unsigned char* output, int width, int height) {
    // Apply a 3x3 averaging filter
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float sum = 0.0f;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int currentX = x + j;
                    int currentY = y + i;

                    // Ensure we're within the image boundaries
                    currentX = std::max(0, std::min(currentX, width - 1));
                    currentY = std::max(0, std::min(currentY, height - 1));

                    // Access the pixel value from the input image
                    unsigned char pixelValue = input[currentY * width + currentX];

                    // Accumulate the pixel values
                    sum += static_cast<float>(pixelValue);
                }
            }

            // Compute the average and set the output pixel value
            output[y * width + x] = static_cast<unsigned char>(sum / 9.0f);
        }
    }
}

int main() {
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
    unsigned char* output_h_cpu = (unsigned char*)malloc(sizeof(unsigned char) * width * height);
    unsigned char* input_d, *output_d;

    // CUDA Timing
    cudaEvent_t startCuda, stopCuda;
    cudaEventCreate(&startCuda);
    cudaEventCreate(&stopCuda);

    cudaMalloc((void**)&input_d, sizeof(unsigned char) * width * height);
    cudaMalloc((void**)&output_d, sizeof(unsigned char) * width * height);

    // Copy input image data to device
    cudaMemcpy(input_d, input_h, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);

    // CUDA Timing: start
    cudaEventRecord(startCuda);

    // Perform image smoothing using CUDA
    imageSmoothing(input_d, output_d, width, height);

    // CUDA Timing: stop
    cudaEventRecord(stopCuda);
    cudaEventSynchronize(stopCuda);

    // Copy result back to host
    unsigned char* output_h_cuda = (unsigned char*)malloc(sizeof(unsigned char) * width * height);
    cudaMemcpy(output_h_cuda, output_d, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

    // Save the input and output images
    cv::imwrite("input_image.jpg", image);
    cv::imwrite("output_smoothed_image_cuda.jpg", cv::Mat(height, width, CV_8UC1, output_h_cuda));

    // Free allocated memory
    free(output_h_cuda);
    cudaFree(input_d);
    cudaFree(output_d);

    // CUDA Timing: calculate and print the time
    float cudaTime = 0.0f;
    cudaEventElapsedTime(&cudaTime, startCuda, stopCuda);
    std::cout << "CUDA Time: " << cudaTime << " ms" << std::endl;

    // CPU Timing
    auto startCpu = std::chrono::high_resolution_clock::now();

    // Perform image smoothing using CPU
    imageSmoothingCPU(input_h, output_h_cpu, width, height);

    auto stopCpu = std::chrono::high_resolution_clock::now();
    auto cpuDuration = std::chrono::duration_cast<std::chrono::milliseconds>(stopCpu - startCpu);

    std::cout << "CPU Time: " << cpuDuration.count() << " ms" << std::endl;

    // Save the output image for CPU
    cv::imwrite("output_smoothed_image_cpu.jpg", cv::Mat(height, width, CV_8UC1, output_h_cpu));

    // Free allocated memory
    free(output_h_cpu);

    return 0;
}


