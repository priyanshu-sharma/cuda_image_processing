// kernel.cu
#include <cuda_runtime.h>
#include <uchar.h> 

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define BLOCK_SIZE 16

__global__ void sobelEdgeDetection(const unsigned char* input, unsigned char* output, int width, int height)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Sobel operator for gradient computation
        int gx = input[(y - 1) * width + (x + 1)] - input[(y - 1) * width + (x - 1)]
                 + 2 * input[y * width + (x + 1)] - 2 * input[y * width + (x - 1)]
                 + input[(y + 1) * width + (x + 1)] - input[(y + 1) * width + (x - 1)];

        int gy = input[(y - 1) * width + (x - 1)] + 2 * input[(y - 1) * width + x] + input[(y - 1) * width + (x + 1)]
                 - input[(y + 1) * width + (x - 1)] - 2 * input[(y + 1) * width + x] - input[(y + 1) * width + (x + 1)];

        // Compute gradient magnitude
        float gradientMagnitude = sqrt(static_cast<float>(gx * gx + gy * gy));

        // Apply thresholding
        output[y * width + x] = (gradientMagnitude > 30) ? 255 : 0;
    }
    else {
        // Border pixels - set to 0
        output[y * width + x] = 0;
    }
}

void edgeDetection(const unsigned char* input, unsigned char* output, int width, int height)
{
    dim3 DimGrid((width - 1) / BLOCK_SIZE + 1, (height - 1) / BLOCK_SIZE + 1, 1);
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    sobelEdgeDetection<<<DimGrid, DimBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
