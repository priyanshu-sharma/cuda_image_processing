#include <cuda_runtime.h>

__global__ void sharpen_kernel(unsigned char *input, unsigned char *output, int width, int height, int stride, int channels) {
    // Calculate the pixel's location
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // An example sharpening filter kernel
    float filter[3][3] = {
        { -1, -1, -1 },
        { -1,  9, -1 },
        { -1, -1, -1 }
    };

    int filterSize = 3;
    int filterHalf = filterSize / 2;

    // Apply the filter to each channel
    for (int c = 0; c < channels; c++) {
        float pixelValue = 0.0;
        // Only execute for valid pixels
        if (x >= filterHalf && y >= filterHalf && x < (width - filterHalf) && y < (height - filterHalf)) {
            for (int filterY = -filterHalf; filterY <= filterHalf; filterY++) {
                for (int filterX = -filterHalf; filterX <= filterHalf; filterX++) {
                    // Find the global image position
                    int imageX = x + filterX;
                    int imageY = y + filterY;
                    
                    // Get the current pixel value
                    unsigned char pixel = input[(imageY * stride) + (imageX * channels) + c];
                    
                    // Apply the filter kernel
                    pixelValue += pixel * filter[filterY + filterHalf][filterX + filterHalf];
                }
            }

            // Write the new pixel value to the output image
            output[(y * stride) + (x * channels) + c] = min(max(int(pixelValue), 0), 255);
        }
    }
}

void sharpen_image(unsigned char *input, unsigned char *output, int width, int height, int channels) {
    unsigned char *dev_input, *dev_output;
    int stride = width * channels; // Adjusted stride for color images

    // Allocate device memory
    cudaMalloc((void **)&dev_input, width * height * channels * sizeof(unsigned char));
    cudaMalloc((void **)&dev_output, width * height * channels * sizeof(unsigned char));

    // Copy input image to device
    cudaMemcpy(dev_input, input, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Setup the execution configuration
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    // Launch the kernel
    sharpen_kernel<<<dimGrid, dimBlock>>>(dev_input, dev_output, width, height, stride, channels);

    // Copy the output image back to the host
    cudaMemcpy(output, dev_output, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_input);
    cudaFree(dev_output);
}
