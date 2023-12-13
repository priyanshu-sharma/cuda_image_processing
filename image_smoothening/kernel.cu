#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define BLOCK_SIZE 16

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

__device__ float applyGaussianFilter(const unsigned char* input, int width, int x, int y)
{
    // Gaussian filter weights
    // const float weights[3][3] = {
    //     {1.0f, 2.0f, 1.0f},
    //     {2.0f, 4.0f, 2.0f},
    //     {1.0f, 2.0f, 1.0f}
    // };

    const float weights[3][3] = {
    {1.0f, 4.0f, 1.0f},
    {4.0f, 16.0f, 4.0f},
    {1.0f, 4.0f, 1.0f}
};

    float sum = 0.0f;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            int currentX = x + i;
            int currentY = y + j;

            // Ensure we're within the image boundaries
            currentX = max(0, min(currentX, width - 1));
            currentY = max(0, min(currentY, width - 1));

            // Access the pixel value from the input image
            unsigned char pixelValue = input[currentY * width + currentX];

            // Accumulate the weighted pixel values
            sum += weights[i + 1][j + 1] * static_cast<float>(pixelValue);
        }
    }

    // Normalize the result
    return sum / 16.0f; // Sum of weights is 16 in this case
}

__global__ void imageSmoothingKernel(const unsigned char* input, unsigned char* output, int width, int height)
{
    // int x = threadIdx.x + blockDim.x * blockIdx.x;
    // int y = threadIdx.y + blockDim.y * blockIdx.y;

    // if (x < width && y < height) {
    //     output[y * width + x] = applyAveragingFilter(input, width, x, y);
    // }

     int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < width && y < height) {
        output[y * width + x] = static_cast<unsigned char>(applyGaussianFilter(input, width, x, y));
    }
}

void imageSmoothing(const unsigned char* input, unsigned char* output, int width, int height)
{
    dim3 DimGrid((width - 1) / BLOCK_SIZE + 1, (height - 1) / BLOCK_SIZE + 1, 1);
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    imageSmoothingKernel<<<DimGrid, DimBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
