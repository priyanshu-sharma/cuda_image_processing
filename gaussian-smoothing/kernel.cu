#include <stdio.h>
#include <math.h>

#define KERNEL_RADIUS 200  // This will create a 7x7 kernel
#define KERNEL_SIZE (2 * KERNEL_RADIUS + 1)
#define BLOCK_SIZE 16
#define SIGMA 50.0 // Adjust sigma value as needed for the blur effect.

// Function to generate a Gaussian kernel.
void generateGaussianKernel(float *kernel, int radius, float sigma) {
    float sum = 0.0;
    for (int row = -radius; row <= radius; row++) {
        for (int col = -radius; col <= radius; col++) {
            float value = expf(-(row*row + col*col) / (2*sigma*sigma));
            kernel[(row + radius) * KERNEL_SIZE + (col + radius)] = value;
            sum += value;
        }
    }
    // Normalize the kernel
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) {
        kernel[i] /= sum;
    }
}

__global__ void gaussianSmoothing(unsigned char *input, unsigned char *output, int width, int height, float *kernel) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col < width && row < height) {
        float sum = 0.0;
        for (int kernelRow = -KERNEL_RADIUS; kernelRow <= KERNEL_RADIUS; kernelRow++) {
            for (int kernelCol = -KERNEL_RADIUS; kernelCol <= KERNEL_RADIUS; kernelCol++) {
                int pixelRow = min(max(row + kernelRow, 0), height - 1);
                int pixelCol = min(max(col + kernelCol, 0), width - 1);
                float pixelValue = (float)input[pixelRow * width + pixelCol];
                sum += pixelValue * kernel[(kernelRow + KERNEL_RADIUS) * KERNEL_SIZE + (kernelCol + KERNEL_RADIUS)];
            }
        }
        output[row * width + col] = (unsigned char)sum;
    }
}

// Host function to call the kernel
void gaussianBlur(unsigned char *input, unsigned char *output, int width, int height, float sigma) {
    float kernel[KERNEL_SIZE * KERNEL_SIZE];
    generateGaussianKernel(kernel, KERNEL_RADIUS, sigma);

    // Copy the kernel to constant memory on the device
    float *deviceKernel;
    cudaMalloc((void**)&deviceKernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMemcpy(deviceKernel, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for input and output images on the device
    unsigned char *deviceInput, *deviceOutput;
    cudaMalloc((void**)&deviceInput, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&deviceOutput, width * height * sizeof(unsigned char));

    // Copy the input image to the device
    cudaMemcpy(deviceInput, input, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 DimGrid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Launch the kernel
    gaussianSmoothing<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, width, height, deviceKernel);

    // Copy the output image back to host
    cudaMemcpy(output, deviceOutput, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(deviceKernel);
}