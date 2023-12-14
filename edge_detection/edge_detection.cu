#include <iostream>
#include <opencv2/opencv.hpp>
#include "kernel.cu"


void edgeDetectionCPU(const unsigned char* input, unsigned char* output, int width, int height) {
    // Sobel filter masks
    const int sobelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    const int sobelY[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    // Apply Sobel filter
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int sumX = 0;
            int sumY = 0;

            // Convolution with Sobel filter
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int currentX = x + j;
                    int currentY = y + i;

                    int pixelValue = input[currentY * width + currentX];

                    sumX += pixelValue * sobelX[i + 1][j + 1];
                    sumY += pixelValue * sobelY[i + 1][j + 1];
                }
            }

            // Compute gradient magnitude
            float gradientMagnitude = sqrt(static_cast<float>(sumX * sumX + sumY * sumY));

            // Apply thresholding
            output[y * width + x] = (gradientMagnitude > 30) ? 255 : 0;
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
    unsigned char* output_h_cuda = (unsigned char*)malloc(sizeof(unsigned char) * width * height);
    unsigned char* output_h_cpu = (unsigned char*)malloc(sizeof(unsigned char) * width * height);

    unsigned char *input_d, *output_d;
    cudaMalloc((void**)&input_d, sizeof(unsigned char) * width * height);
    cudaMalloc((void**)&output_d, sizeof(unsigned char) * width * height);

    // Copy input image data to device
    cudaMemcpy(input_d, input_h, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);

    // CUDA Timing
    cudaEvent_t startCuda, stopCuda;
    cudaEventCreate(&startCuda);
    cudaEventCreate(&stopCuda);

    cudaEventRecord(startCuda);

    // Perform edge detection using CUDA
    edgeDetection(input_d, output_d, width, height);

    cudaEventRecord(stopCuda);
    cudaEventSynchronize(stopCuda);

    float cudaTime;
    cudaEventElapsedTime(&cudaTime, startCuda, stopCuda);
    std::cout << "CUDA Time: " << cudaTime << " ms" << std::endl;

    // Copy the CUDA result back to the host
    cudaMemcpy(output_h_cuda, output_d, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

    // CPU Timing
    int64_t startCpu = cv::getTickCount();

    // Perform edge detection using CPU
    edgeDetectionCPU(input_h, output_h_cpu, width, height);

    int64_t endCpu = cv::getTickCount();
    double cpuTime = (endCpu - startCpu) / cv::getTickFrequency() * 1000.0;
    std::cout << "CPU Time: " << cpuTime << " ms" << std::endl;

    // Validate that CUDA and CPU implementations produce the same result
    for (int i = 0; i < width * height; ++i) {
        if (output_h_cuda[i] != output_h_cpu[i]) {
            std::cerr << "Verification failed: CUDA and CPU results differ." << std::endl;
            break;
        }
    }

    // Save the input and output images
    cv::imwrite("input_image.jpg", image);
    cv::imwrite("output_edge_detected_cuda.jpg", cv::Mat(height, width, CV_8UC1, output_h_cuda));
    cv::imwrite("output_edge_detected_cpu.jpg", cv::Mat(height, width, CV_8UC1, output_h_cpu));

    // Free allocated memory
    free(output_h_cuda);
    free(output_h_cpu);
    cudaFree(input_d);
    cudaFree(output_d);

    return 0;
}
