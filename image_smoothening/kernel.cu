#include <cuda_runtime.h>
# define BLOCK_SIZE 512

__device__ void applyAveragingFilter(const unsigned char* input, unsigned char* output, int width, int height, int x, int y)
{
    // Apply a 3x3 averaging filter
    float sum = 0.0f;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            int currentX = x + i;
            int currentY = y + j;

            // Ensure we're within the image boundaries
            currentX = max(0, min(currentX, width - 1));
            currentY = max(0, min(currentY, height - 1));

            // Access the pixel value from the input image
            unsigned char pixelValue = input[currentY * width + currentX];

            // Accumulate the pixel values
            sum += static_cast<float>(pixelValue);
        }
    }

    // Compute the average and store the result in the output image
    output[y * width + x] = static_cast<unsigned char>(sum / 9.0f);
}

__global__ void smoothening_kernel(const unsigned char* input, unsigned char* output, int width, int height)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < width && y < height) {
        applyAveragingFilter(input, output, width, height, x, y);
    }
}

void smoothening(const unsigned char* input, unsigned char* output, int width, int height) {
    dim3 DimGrid((width - 1) / BLOCK_SIZE + 1, (height - 1) / BLOCK_SIZE + 1, 1);
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    smoothening_kernel<<<DimGrid, DimBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
