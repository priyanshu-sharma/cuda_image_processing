#include <stdio.h>
#include <stdint.h>
// #include <opencv2/opencv.hpp>
// #include <iostream>

// // Kernel function definition
// #include "kernel.cu"

// using namespace cv;
// using namespace std;

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "kernel.cu"  // Include your CUDA kernel file

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    cudaError_t cuda_ret;
    int *input_h, *output_h;
    int *input_d, *output_d;

    Mat image = imread("demo.png", IMREAD_GRAYSCALE);
    if (!image.data) {
        cout << "Could not open or find the image." << endl;
        return -1;
    }

    unsigned char *myData = (unsigned char*)(image.data);
    // int width = image.cols;
    // int height = image.rows;
    // int stride = image.step;
    // int image_size = width * height;
    // cout << "Image Size=" << image_size << endl;
    // cout << "Width=" << width << endl;
    // cout << "Height=" << height << endl;
    // cout << "Stride=" << stride << endl;

    int width = image.cols;
    int height = image.rows;
    int stride = image.step;
    int image_size = width * height;
    cout<<"Image Size="<<image_size<<endl;
    cout<<"Width="<<width<<endl;
    cout<<"Height="<<height<<endl;
    cout<<"Stride="<<stride<<endl;

    input_h = (int *) malloc(sizeof(int) * image_size);
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            unsigned char val = myData[ i * stride + j];
            input_h[i * stride + j] = int(val);
        }
    }
    output_h = (int *) malloc(sizeof(int) * image_size);


    // input_h = (unsigned char*)malloc(sizeof(unsigned char) * image_size);
    // for (int i = 0; i < height; i++) {
    //     for (int j = 0; j < width; j++) {
    //         unsigned char val = myData[i * stride + j];
    //         input_h[i * stride + j] = val;
    //     }
    // }
    // output_h = (unsigned char*)malloc(sizeof(unsigned char) * image_size);

//     input_h = (unsigned char*)malloc(sizeof(unsigned char) * image_size);
// for (int i = 0; i < height; i++) {
//     for (int j = 0; j < width; j++) {
//         unsigned char val = myData[i * stride + j];
//         input_h[i * width + j] = val;  // Use 'width' instead of 'stride'
//     }
// }
// output_h = (unsigned char*)malloc(sizeof(unsigned char) * image_size);

    cudaMalloc((void**)&input_d, sizeof(unsigned char) * image_size);
    cudaMalloc((void**)&output_d, sizeof(unsigned char) * image_size);

     // Check for CUDA errors after memory allocation
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        cerr << "CUDA error after cudaMalloc: " << cudaGetErrorString(cudaError) << endl;

        // Free allocated memory before exiting
        free(input_h);
        free(output_h);
        cudaFree(input_d);
        cudaFree(output_d);

        return -1;
    }
    cudaMemcpy(input_d, input_h, sizeof(unsigned char) * image_size, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    // Define block and grid dimensions
    dim3 blockDim(8, 8);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Call your CUDA kernel for image smoothening
    // smoothening_kernel(input_d, output_d, width, height);
    // / Call your CUDA kernel for image smoothening
    smoothening(input_d, output_d, width, height);

    cuda_ret = cudaGetLastError();
if (cuda_ret != cudaSuccess) {
    cerr << "CUDA error after kernel launch: " << cudaGetErrorString(cuda_ret) << endl;
    // Handle the error appropriately
}      
    
    cuda_ret = cudaDeviceSynchronize();
    // cuda_ret = cudaDeviceSynchronize();
if (cuda_ret != cudaSuccess) {
    cerr << "CUDA error during synchronization: " << cudaGetErrorString(cuda_ret) << endl;
    // return -1;
}


    // Copy device variables from host ----------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    cudaMemcpy(output_h, output_d, sizeof(unsigned char) * image_size, cudaMemcpyDeviceToHost);

    // Create Mat objects for input and output images
    // Mat input_image(height, width, CV_8UC1, input_h);
    // Mat output_image(height, width, CV_8UC1, output_h);
    // Create Mat objects for input and output images
    Mat input_image(height, width, CV_8UC3, input_h);
    Mat output_image(height, width, CV_8UC3, output_h);


    // Save the input and output images
    imwrite("input.jpg", input_image);
    imwrite("output.jpg", output_image);

    // Free allocated memory
    free(input_h);
    free(output_h);
    cudaFree(input_d);
    cudaFree(output_d);

    return 0;
}
