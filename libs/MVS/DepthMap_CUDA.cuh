#include "cuda_runtime.h"  
#include "device_launch_parameters.h" 

struct DepthEstimator_CUDA 
{
	float pi_cuda = 3.1415926535897932384626433832795;
	
	//genreate random number by CUDA
	float RandomRangeCUDA(float min,float max);	
	//degree to radian
	float FromDegreeToRadian(double degree);
	//radian to degree
	float FromRadianToDegree(double radian);

	// generate random depth
	float RandomDepth(float dMin, float dMax);
	//generate random normal
	float* RandomNormal();
	//normal to direction
	void Normal2Dir(float *normal,float *direction);
	//direction to normal
	void Direction2Normal(float alpha,float beta,float *normal);

};