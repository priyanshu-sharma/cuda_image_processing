#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define BLOCK_SIZE 16
__device__ unsigned char applyAveragingFilter(const unsigned char* input, int width, int x, int y)
{
    // Apply a 5x5 averaging filter
    float sum = 0.0f;
    for (int i = -2; i <= 2; ++i) {
        for (int j = -2; j <= 2; ++j) {
            int currentX = x + i;
            int currentY = y + j;

            // Ensure we're within the image boundaries
            currentX = max(0, min(currentX, width - 1));
            currentY = max(0, min(currentY, width - 1));

            // Access the pixel value from the input image
            unsigned char pixelValue = input[currentY * width + currentX];

            // Accumulate the pixel values
            sum += static_cast<float>(pixelValue);
        }
    }

    // Compute the average and return the result
    return static_cast<unsigned char>(sum / 25.0f);
}
// __device__ unsigned char applyAveragingFilter(const unsigned char* input, int width, int x, int y)
// {
//     // Apply a 3x3 averaging filter
//     float sum = 0.0f;
//     for (int i = -1; i <= 1; ++i) {
//         for (int j = -1; j <= 1; ++j) {
//             int currentX = x + i;
//             int currentY = y + j;

//             // Ensure we're within the image boundaries
//             currentX = max(0, min(currentX, width - 1));
//             currentY = max(0, min(currentY, width - 1));

//             // Access the pixel value from the input image
//             unsigned char pixelValue = input[currentY * width + currentX];

//             // Accumulate the pixel values
//             sum += static_cast<float>(pixelValue);
//         }
//     }

//     // Compute the average and return the result
//     return static_cast<unsigned char>(sum / 9.0f);
// }

__global__ void imageBlurKernel(const unsigned char* input, unsigned char* output, int width, int height)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < width && y < height) {
        output[y * width + x] = applyAveragingFilter(input, width, x, y);
    }
}

void imageBlur(const unsigned char* input, unsigned char* output, int width, int height)
{
    dim3 DimGrid((width - 1) / BLOCK_SIZE + 1, (height - 1) / BLOCK_SIZE + 1, 1);
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    imageBlurKernel<<<DimGrid, DimBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}