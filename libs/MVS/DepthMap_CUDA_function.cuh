#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include "curand_kernel.h" 

#include <stdio.h>

//degree to radian
__device__ float FromDegreeToRadian(double degree);
//radian to degree
__device__ float FromRadianToDegree(double radian);
//normal to direction
__device__ void Normal2Dir(float *normal,float *direction);
//direction to normal
__device__ void Direction2Normal(float alpha,float beta,float *normal);


//genreate random number by CUDA
__device__ float RandomRangeCUDA(int seed,float min,float max);	
// generate random depth
__device__ float RandomDepth(int seed,float dMin, float dMax);
//generate random normal
__device__ float* RandomNormal(int seed);


__global__ void test_random_kernel(float *result,float min_value,float max_value);

extern "C"
{
    void test_random();
}