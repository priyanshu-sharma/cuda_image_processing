// kernel.cu
#include <cuda_runtime.h>
#include <uchar.h> 

__global__ void sobelEdgeDetection(const uchar *inputImage, uchar *outputImage, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int gx = inputImage[(y - 1) * width + (x + 1)] + 2 * inputImage[y * width + (x + 1)] + inputImage[(y + 1) * width + (x + 1)] -
                  (inputImage[(y - 1) * width + (x - 1)] + 2 * inputImage[y * width + (x - 1)] + inputImage[(y + 1) * width + (x - 1)]);

        int gy = inputImage[(y + 1) * width + (x - 1)] + 2 * inputImage[(y + 1) * width + x] + inputImage[(y + 1) * width + (x + 1)] -
                  (inputImage[(y - 1) * width + (x - 1)] + 2 * inputImage[(y - 1) * width + x] + inputImage[(y - 1) * width + (x + 1)]);

        outputImage[y * width + x] = static_cast<uchar>(sqrt(gx * gx + gy * gy));
    }
}
