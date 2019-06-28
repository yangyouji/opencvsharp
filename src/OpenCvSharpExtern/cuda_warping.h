#ifndef _CPP_GPU_WARPING_H_
#define _CPP_GPU_WARPING_H_

#ifdef ENABLED_CUDA

#include "include_opencv.h"
using namespace cv::cuda;

CVAPI(void) cuda_imgproc_resize(cv::_InputArray* src, cv::_OutputArray* dst, CvSize dsize, double fx, double fy, int interpolation, Stream* stream)
{
	cv::cuda::resize(*src, *dst, dsize, fx, fy, interpolation, *stream);
}

CVAPI(void) cuda_imgproc_pyrDown(cv::_InputArray *src, cv::_OutputArray *dst, Stream* stream)
{
	cv::cuda::pyrDown(*src, *dst, *stream);
}

CVAPI(void) cuda_imgproc_pyrUp(cv::_InputArray *src, cv::_OutputArray *dst, Stream* stream)
{
	cv::cuda::pyrUp(*src, *dst, *stream);
}

//CVAPI(void) cuda_imgproc_pyrUp(cv::_InputArray *src, cv::_OutputArray *dst, cv::cuda::Stream steam)
//{
//	cv::cuda::pyrUp(*src, *dst, steam);
//}

#endif

#endif