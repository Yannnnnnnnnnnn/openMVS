#include "DepthMap_CUDA.cuh"



// generate random depth
float DepthEstimator_CUDA::RandomDepth(float dMin, float dMax) 
{
    return RandomRangeCUDA(dMin, dMax);
}
//generate random normal
float* DepthEstimator_CUDA::RandomNormal() 
{
    //we must know why we choose 0-360 and 120-180
    //I know
    float a1Min = FromDegreeToRadian(0.f);
    float a1Max = FromDegreeToRadian(360.f);
    float a2Min = FromDegreeToRadian(120.f);
    float a2Max = FromDegreeToRadian(180.f);

    //remember to free this after using
    float *normal = new float(3);

    Direction2Normal(RandomRangeCUDA(a1Min,a1Max),RandomRangeCUDA(a2Min,a2Max),normal);

    return normal;
}

//degree to radian
float DepthEstimator_CUDA::FromDegreeToRadian(double degree)
{
    return ((degree)*(pi_cuda/180.f));
}

//radian to degree
float DepthEstimator_CUDA::FromRadianToDegree(double radian)
{
    return ((radian)*(180.f/pi_cuda));
}

//normal to direction
void DepthEstimator_CUDA::Normal2Dir(float *normal,float *direction) 
{
    direction[1] = atan2(sqrt(normal[0]*normal[0] + normal[1]*normal[1]), normal[2]);
    direction[0] = atan2(normal[1],normal[0]);
}

//direction to normal
void DepthEstimator_CUDA::Direction2Normal(float alpha,float beta,float *normal) 
{
    float siny = sin(beta);
    normal[0] = cos(alpha)*siny;
    normal[1] = sin(alpha)*siny;
    normal[2] = cos(beta);
}

//genreate random number by CUDA
float RandomRangeCUDA(float min,float max)
{
    
}