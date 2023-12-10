#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char* argv[])
{
    printf("Working");
    Mat image = imread("demo.png", IMREAD_GRAYSCALE);
    if (!image.data) { 
        printf("No image data \n");  
    }
    else{
        printf("data\n");
    }
    return 0;
}