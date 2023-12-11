// main.cu
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat inputImage = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);

    if (inputImage.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    size_t imageSize = width * height * sizeof(uchar);

    uchar *d_inputImage, *d_outputImage;

    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);

    cudaMemcpy(d_inputImage, inputImage.data, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    sobelEdgeDetection<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);

    uchar *outputImage = new uchar[imageSize];
    cudaMemcpy(outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    cv::Mat outputMat(height, width, CV_8U, outputImage);
    cv::imwrite("output.jpg", outputMat);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    delete[] outputImage;

    return 0;
}
