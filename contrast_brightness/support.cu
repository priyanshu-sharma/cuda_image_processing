#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

// void verify(float *A, float *B, float *C, unsigned int dim) {

//   const float relativeTolerance = 1e-6;
//   unsigned int count = 0;

//   for(int row = 0; row < dim; ++row) {
//     count += C[row*dim];
// 	for(int col = 0; col < dim; ++col) {
//       float sum = A[row*dim + col] + B[row*dim + col]; 
//       float relativeError = (sum - C[row*dim + col])/sum;
// 	  if (relativeError > relativeTolerance
//         || relativeError < -relativeTolerance) {
//         printf("\nTEST FAILED %u\n\n",count);
//         exit(1);
//       }
//     }
//   }
//   printf("TEST PASSED %u\n\n", count);

// }

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

