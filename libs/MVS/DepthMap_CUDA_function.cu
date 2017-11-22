#include "DepthMap_CUDA_function.cuh"


//degree to radian
__device__ float FromDegreeToRadian(double degree)
{
    return ((degree)*(3.1415926535897932384626433832795/180.f));
}

//radian to degree
__device__ float FromRadianToDegree(double radian)
{
    return ((radian)*(180.f/3.1415926535897932384626433832795));
}

//normal to direction
__device__ void Normal2Dir(float *normal,float *direction) 
{
    direction[1] = atan2(sqrt(normal[0]*normal[0] + normal[1]*normal[1]), normal[2]);
    direction[0] = atan2(normal[1],normal[0]);
}

//direction to normal
__device__ void Direction2Normal(float alpha,float beta,float *normal) 
{
    float siny = sin(beta);
    normal[0] = cos(alpha)*siny;
    normal[1] = sin(alpha)*siny;
    normal[2] = cos(beta);
}

//genreate random number by CUDA
__device__ float RandomRangeCUDA(int seed,float min,float max)
{
    curandState state;
    clock_t current_time = clock(); 
    curand_init(seed, current_time, 0, &state);
    float random_float = abs(curand_uniform_double(&state));

    return (max-min)*random_float+min;
}

// generate random depth
__device__ float RandomDepth(int seed,float dMin, float dMax) 
{
    return RandomRangeCUDA(seed,dMin,dMax);
}

//generate random normal
__device__ float* RandomNormal(int seed) 
{
    //we must know why we choose 0-360 and 120-180
    //I know
    float a1Min = FromDegreeToRadian(0.f);
    float a1Max = FromDegreeToRadian(360.f);
    float a2Min = FromDegreeToRadian(120.f);
    float a2Max = FromDegreeToRadian(180.f);

    //remember to free this after using
    float *normal = new float(3);

    Direction2Normal(RandomRangeCUDA(seed,a1Min,a1Max),RandomRangeCUDA(seed,a2Min,a2Max),normal);

    return normal;
}


__global__ void test_random_kernel(float *result,float min_value,float max_value)
{
    int id = threadIdx.x;

    result[id]=RandomRangeCUDA(id,min_value,max_value);
}

extern "C"
void test_random()
{
    float min_value = 10.5;
    float max_value = 100.20;

    float random_value[100];

    float *dev_result=NULL;

    // Allocate space for results on device 
    cudaMalloc((void**)&dev_result,100*sizeof(float));
    cudaMemset(dev_result,0,100*sizeof(float));

    test_random_kernel<<<1,100>> >(dev_result,min_value,max_value);

   //Copy device memory to host 
    cudaMemcpy(&random_value,dev_result,100*sizeof(float),cudaMemcpyDeviceToHost);
    
    /* Show result */
    for(int i=0;i<100;i++)
    {
        printf("%lf \n",random_value[i]);
    }

}