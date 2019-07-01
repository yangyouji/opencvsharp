#ifndef _CPP_GPU_WARPING_H_
#define _CPP_GPU_WARPING_H_

#ifdef ENABLED_CUDA

#include "include_opencv.h"
using namespace cv::cuda;

CVAPI(void) cuda_warping_remap(cv::_InputArray* src, cv::_OutputArray* dst, cv::_InputArray* map1, cv::_InputArray* map2,
	int interpolation, int borderMode, CvScalar borderValue, Stream* stream)
{
	cv::cuda::remap(*src, *dst, *map1, *map2, interpolation, borderMode, borderValue, *stream);
}

CVAPI(void) cuda_warping_resize(cv::_InputArray* src, cv::_OutputArray* dst, CvSize dsize, double fx, double fy, int interpolation, Stream* stream)
{
	cv::cuda::resize(*src, *dst, dsize, fx, fy, interpolation, *stream);
}

CVAPI(void) cuda_warping_pyrDown(cv::_InputArray* src, cv::_OutputArray* dst, Stream* stream)
{
	cv::cuda::pyrDown(*src, *dst, *stream);
}

CVAPI(void) cuda_warping_pyrUp(cv::_InputArray* src, cv::_OutputArray* dst, Stream* stream)
{
	cv::cuda::pyrUp(*src, *dst, *stream);
}

CVAPI(void) cuda_warping_warpAffine(cv::_InputArray* src, cv::_OutputArray* dst, cv::_InputArray* M, CvSize dsize,
	int flags, int borderMode, CvScalar borderValue, Stream* stream)
{
	cv::cuda::warpAffine(*src, *dst, *M, dsize, flags, borderMode, borderValue, *stream);
}

CVAPI(void) cuda_warping_buildWarpAffineMaps(cv::_InputArray* M, bool inverse, CvSize dsize, cv::_OutputArray* xmap
	, cv::_OutputArray* ymap, Stream* stream)
{
	cv::cuda::buildWarpAffineMaps(*M, inverse, dsize, *xmap, *ymap, *stream);
}

CVAPI(void) cuda_warping_warpPerspective(cv::_InputArray* src, cv::_OutputArray* dst, cv::_InputArray* m, CvSize dsize,
	int flags, int borderMode, CvScalar borderValue, Stream* stream)
{
	cv::cuda::warpPerspective(*src, *dst, *m, dsize, flags, borderMode, borderValue, *stream);
}

CVAPI(void) cuda_warping_buildWarpPerspectiveMaps(cv::_InputArray* M, bool inverse, CvSize dsize, cv::_OutputArray* xmap
	, cv::_OutputArray* ymap, Stream* stream)
{
	cv::cuda::buildWarpPerspectiveMaps(*M, inverse, dsize, *xmap, *ymap, *stream);
}

CVAPI(void) cuda_warping_rotate(cv::_InputArray* src, cv::_OutputArray* dst, CvSize dsize, double angle,
	double xShift, double yShift, int flags, Stream* stream)
{
	cv::cuda::rotate(*src, *dst, dsize, angle, xShift, yShift, flags, *stream);
}

#endif

#endif