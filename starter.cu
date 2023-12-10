#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
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
    cout<<"Width="<<unsigned(width)<<endl;
    // printf("Width");
    // snprintf(myString, sizeof(myString), "%u", width);
    // snprintf(myString, sizeof(myString), "%u", height);
    // snprintf(myString, sizeof(myString), "%u", _stride);
    // printf(width, height, _stride);
    return 0;
}