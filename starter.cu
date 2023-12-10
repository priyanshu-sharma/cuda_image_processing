#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
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
    // int count = 0;
    vector<int> image_vector;
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            uint8_t val = myData[ i * _stride + j];
            // cout<<"Image at : "<<i<<" , "<<j<<" - "<<unsigned(val)<<endl;
            // count = count + 1;
            image_vector.push_back(unsigned(val));
        }
    }
    cout<<"Count="<<count<<endl;
    cout<<"Size="<<image_vector.size()<<endl;
    cout<<"Width="<<unsigned(width)<<endl;
    cout<<"Height="<<unsigned(height)<<endl;
    cout<<"Stride="<<unsigned(_stride)<<endl;
    return 0;
}