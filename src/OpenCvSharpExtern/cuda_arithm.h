#ifndef _CPP_GPU_ARITHM_H_
#define _CPP_GPU_ARITHM_H_

#ifdef ENABLED_CUDA

#include "include_opencv.h"
using namespace cv::cuda;

CVAPI(int) cuda_countNonZero(cv::_InputArray* src)
{
	return cv::cuda::countNonZero(*src);
}

#endif

#endif
