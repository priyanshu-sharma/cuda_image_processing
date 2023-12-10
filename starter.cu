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
    uint8_t *myData = image.data;
    int width = image.cols;
    int height = image.rows;
    int _stride = image.step;
    char myString[5];
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            uint8_t val = myData[ i * _stride + j];
            snprintf(myString, sizeof(myString), "%u", val);
        }
    }
    printf("Width");
    printf(width, height, _stride);
    return 0;
}