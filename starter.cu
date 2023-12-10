#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char* argv[])
{
    printf("Working");
    Mat image = imread("demo.png", IMREAD_GRAYSCALE);
    printf("Image data");
    printf(image.data);
    return 0;
}