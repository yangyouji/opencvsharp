#ifndef _CPP_GPU_ARITHM_H_
#define _CPP_GPU_ARITHM_H_

#ifdef ENABLED_CUDA

#include "include_opencv.h"
using namespace cv::cuda;


CVAPI(void) cuda_arithm_add(cv::_InputArray* src1, cv::_InputArray* src2, cv::_OutputArray* dst, cv::_InputArray* mask, int dtype, Stream* stream)
{
	if (mask == nullptr)
		cv::cuda::add(*src1, *src2, *dst, cv::noArray(), dtype, *stream);
	else
		cv::cuda::add(*src1, *src2, *dst, *mask, dtype, *stream);
}

CVAPI(void) cuda_arithm_subtract(cv::_InputArray* src1, cv::_InputArray* src2, cv::_OutputArray* dst, cv::_InputArray* mask, int dtype, Stream* stream)
{
	if (mask == nullptr)
		cv::cuda::subtract(*src1, *src2, *dst, cv::noArray(), dtype, *stream);
	else
		cv::cuda::subtract(*src1, *src2, *dst, *mask, dtype, *stream);
}

CVAPI(void) cuda_arithm_multiply(cv::_InputArray* src1, cv::_InputArray* src2, cv::_OutputArray* dst, double scale, int dtype, Stream* stream)
{
	cv::cuda::multiply(*src1, *src2, *dst, scale, dtype, *stream);
}

CVAPI(void) cuda_arithm_divide(cv::_InputArray* src1, cv::_InputArray* src2, cv::_OutputArray* dst, double scale, int dtype, Stream* stream)
{
	cv::cuda::divide(*src1, *src2, *dst, scale, dtype, *stream);
}

CVAPI(void) cuda_arithm_absdiff(cv::_InputArray* src1, cv::_InputArray* src2, cv::_OutputArray* dst, Stream* stream)
{
	cv::cuda::absdiff(*src1, *src2, *dst, *stream);
}

CVAPI(void) cuda_arithm_abs(cv::_InputArray* src, cv::_OutputArray* dst, Stream* stream)
{
	cv::cuda::abs(*src, *dst, *stream);
}

CVAPI(void) cuda_arithm_sqr(cv::_InputArray* src, cv::_OutputArray* dst, Stream* stream)
{
	cv::cuda::sqr(*src, *dst, *stream);
}

CVAPI(void) cuda_arithm_sqrt(cv::_InputArray* src, cv::_OutputArray* dst, Stream* stream)
{
	cv::cuda::sqrt(*src, *dst, *stream);
}

CVAPI(void) cuda_arithm_exp(cv::_InputArray* src, cv::_OutputArray* dst, Stream* stream)
{
	cv::cuda::exp(*src, *dst, *stream);
}

CVAPI(void) cuda_arithm_log(cv::_InputArray* src, cv::_OutputArray* dst, Stream* stream)
{
	cv::cuda::log(*src, *dst, *stream);
}

CVAPI(void) cuda_arithm_pow(cv::_InputArray* src, double power, cv::_OutputArray* dst, Stream* stream)
{
	cv::cuda::pow(*src, power, *dst, *stream);
}

CVAPI(void) cuda_arithm_compare(cv::_InputArray* src1, cv::_InputArray* src2, cv::_OutputArray* dst, int cmpop, Stream* stream)
{
	cv::cuda::compare(*src1, *src2, *dst, cmpop, *stream);
}

CVAPI(void) cuda_arithm_bitwise_not(cv::_InputArray* src, cv::_OutputArray* dst, cv::_InputArray* mask, Stream* stream)
{
	cv::cuda::bitwise_not(*src, *dst, entity(mask), *stream);
}

CVAPI(void) cuda_arithm_bitwise_or(cv::_InputArray* src1, cv::_InputArray* src2, cv::_OutputArray* dst, cv::_InputArray* mask, Stream* stream)
{
	cv::cuda::bitwise_or(*src1, *src2, *dst, entity(mask), *stream);
}

CVAPI(void) cuda_arithm_bitwise_and(cv::_InputArray* src1, cv::_InputArray* src2, cv::_OutputArray* dst, cv::_InputArray* mask, Stream* stream)
{
	cv::cuda::bitwise_and(*src1, *src2, *dst, entity(mask), *stream);
}

CVAPI(void) cuda_arithm_bitwise_xor(cv::_InputArray* src1, cv::_InputArray* src2, cv::_OutputArray* dst, cv::_InputArray* mask, Stream* stream)
{
	cv::cuda::bitwise_xor(*src1, *src2, *dst, entity(mask), *stream);
}

CVAPI(void) cuda_arithm_min(cv::_InputArray* src1, cv::_InputArray* src2, cv::_OutputArray* dst, Stream* stream)
{
	cv::cuda::min(*src1, *src2, *dst, *stream);
}

CVAPI(void) cuda_arithm_max(cv::_InputArray* src1, cv::_InputArray* src2, cv::_OutputArray* dst, Stream* stream)
{
	cv::cuda::max(*src1, *src2, *dst, *stream);
}

CVAPI(void) cuda_arithm_addWeighted(cv::_InputArray* src1, double alpha, cv::_InputArray* src2, double beta, double gamma, cv::_OutputArray* dst,
	int dtype, Stream* stream)
{
	cv::cuda::addWeighted(*src1, alpha, *src2, beta, gamma, *dst, dtype, *stream);
}

CVAPI(void) cuda_arithm_threshold(cv::_InputArray* src, cv::_OutputArray* dst, double thresh, double maxval, int type, Stream* stream)
{
	cv::cuda::threshold(*src, *dst, thresh, maxval, type, *stream);
}

CVAPI(void) cuda_arithm_magnitude_0(cv::_InputArray* xy, cv::_OutputArray* magnitude, Stream* stream)
{
	cv::cuda::magnitude(*xy, *magnitude, *stream);
}

CVAPI(void) cuda_arithm_magnitudeSqr_0(cv::_InputArray* xy, cv::_OutputArray* magnitude, Stream* stream)
{
	cv::cuda::magnitudeSqr(*xy, *magnitude, *stream);
}

CVAPI(void) cuda_arithm_magnitude_1(cv::_InputArray* x, cv::_InputArray* y, cv::_OutputArray* magnitude, Stream* stream)
{
	cv::cuda::magnitude(*x, *y, *magnitude, *stream);
}

CVAPI(void) cuda_arithm_magnitudeSqr_1(cv::_InputArray* x, cv::_InputArray* y, cv::_OutputArray* magnitude, Stream* stream)
{
	cv::cuda::magnitudeSqr(*x, *y, *magnitude, *stream);
}






CVAPI(int) cuda_arithm_countNonZero(cv::_InputArray* src)
{
	return cv::cuda::countNonZero(*src);
}

#endif

#endif
