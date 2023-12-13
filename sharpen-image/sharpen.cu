#include <iostream>
#include <opencv2/opencv.hpp>
#include "kernel.cu"

using namespace cv;
using namespace std;

// Forward declaration of the sharpening function
void sharpen_image(unsigned char *input, unsigned char *output, int width, int height, int channels);

int main(int argc, char* argv[])
{
    const char* inputFile = "demo.png";
    const char* outputFile = "sharpen_image.png";

    // Read the image in color
    Mat image = imread(inputFile, IMREAD_COLOR);
    if (image.empty()) { 
        cerr << "Could not open or find the image\n";
        return -1;
    }

    // Ensure the image is in the expected format
    if(image.type() != CV_8UC3) {
        cerr << "Image format must be 8-bit, 3 channels.\n";
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();
    int stride = image.step;
    unsigned int image_size = width * height * channels;

    // Allocate memory for input and output images on host
    unsigned char *input_h, *output_h;
    input_h = (unsigned char *)malloc(image_size * sizeof(unsigned char));
    output_h = (unsigned char *)malloc(image_size * sizeof(unsigned char));

    // Copy image data from OpenCV input image to the input array
    memcpy(input_h, image.data, image_size);

    // Call the sharpen function
    sharpen_image(input_h, output_h, width, height, channels);

    // Create an empty Mat to store the result
    Mat result(height, width, CV_8UC3, output_h);
    // Save the result to a file
    imwrite(outputFile, result);

    // Free host memory
    free(input_h);
    free(output_h);

    cout << "Image sharpening completed successfully." << endl;

    return 0;
}