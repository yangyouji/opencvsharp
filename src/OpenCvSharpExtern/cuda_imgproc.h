#ifndef _CPP_GPU_IMGPROC_H_
#define _CPP_GPU_IMGPROC_H_

#ifdef ENABLED_CUDA

#include "include_opencv.h"
using namespace cv::cuda;


CVAPI(cv::Ptr<CannyEdgeDetector>*) cuda_createCannyEdgeDetector(double low_thresh, double high_thresh, int apperture_size = 3, bool L2gradient = false)
{
	cv::Ptr<CannyEdgeDetector> ptr = cv::cuda::createCannyEdgeDetector(low_thresh, high_thresh, apperture_size, L2gradient);
	return new cv::Ptr<CannyEdgeDetector>(ptr);
}

CVAPI(void) cuda_CannyEdgeDetector_detect(CannyEdgeDetector *obj, cv::_InputArray *image, cv::_OutputArray *edges, Stream* stream)
{
	obj->detect(*image, *edges, *stream);
}

CVAPI(void) cuda_Ptr_CannyEdgeDetector_delete(cv::Ptr<CannyEdgeDetector> *obj)
{
	delete obj;
}

CVAPI(CannyEdgeDetector*) cuda_Ptr_CannyEdgeDetector_get(
	cv::Ptr<CannyEdgeDetector> *ptr)
{
	return ptr->get();
}


#endif

#endif