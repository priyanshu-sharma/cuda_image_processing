#include <opencv2/opencv.hpp>
#include <iostream>
#include "kernel.cu"

#define BLOCK_SIZE 16

// CPU implementation of image blur using a 5x5 averaging filter
void imageBlurCPU(const unsigned char* input, unsigned char* output, int width, int height) {
    for (int y = 2; y < height - 2; ++y) {
        for (int x = 2; x < width - 2; ++x) {
            float sum = 0.0f;
            for (int i = -2; i <= 2; ++i) {
                for (int j = -2; j <= 2; ++j) {
                    int currentX = x + i;
                    int currentY = y + j;
                    unsigned char pixelValue = input[currentY * width + currentX];
                    sum += static_cast<float>(pixelValue);
                }
            }
            output[y * width + x] = static_cast<unsigned char>(sum / 25.0f);
        }
    }
}

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
    unsigned char* output_h_cuda = (unsigned char*)malloc(sizeof(unsigned char) * width * height);
    unsigned char* output_h_cpu = (unsigned char*)malloc(sizeof(unsigned char) * width * height);
    unsigned char* input_d, *output_d;

    // Allocate memory on the GPU
    cudaMalloc((void**)&input_d, sizeof(unsigned char) * width * height);
    cudaMalloc((void**)&output_d, sizeof(unsigned char) * width * height);

    // Copy input image data to the GPU
    cudaMemcpy(input_d, input_h, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);

    // CUDA Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Apply image blur using CUDA
    imageBlur(input_d, output_d, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cudaTime;
    cudaEventElapsedTime(&cudaTime, start, stop);
    std::cout << "CUDA Time: " << cudaTime << " ms" << std::endl;

    // Copy the CUDA result back to the host
    cudaMemcpy(output_h_cuda, output_d, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

    // CPU Timing
    int64_t cpuStart = cv::getTickCount();

    // Apply image blur using CPU
    imageBlurCPU(input_h, output_h_cpu, width, height);

    int64_t cpuEnd = cv::getTickCount();
    double cpuTime = (cpuEnd - cpuStart) / cv::getTickFrequency() * 1000.0;
    std::cout << "CPU Time: " << cpuTime << " ms" << std::endl;

    // Validate that CUDA and CPU implementations produce the same result
    for (int i = 0; i < width * height; ++i) {
        if (output_h_cuda[i] != output_h_cpu[i]) {
            std::cerr << "Verification failed: CUDA and CPU results differ." << std::endl;
            break;
        }
    }

    // Create a Mat object for the output image
    cv::Mat output_image_cuda(height, width, CV_8UC1, output_h_cuda);

    // Save the CUDA output image
    cv::imwrite("output_blurred_cuda.jpg", output_image_cuda);

    // Free allocated memory
    free(output_h_cuda);
    free(output_h_cpu);
    cudaFree(input_d);
    cudaFree(output_d);

    return 0;
}
