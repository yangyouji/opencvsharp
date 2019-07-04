#ifndef _CPP_GPU_IMGPROC_H_
#define _CPP_GPU_IMGPROC_H_

#ifdef ENABLED_CUDA

#include "include_opencv.h"
using namespace cv::cuda;


CVAPI(void) cuda_imgproc_demosaicing(cv::_InputArray* src, cv::_OutputArray* dst, int code, int dcn, Stream* stream) {
	cv::cuda::demosaicing(*src, *dst, code, dcn, *stream);
}

CVAPI(void) cuda_imgproc_swapChannels(cv::_InputOutputArray* src, int dstOrder[4], Stream* stream) {
	cv::cuda::swapChannels(*src, dstOrder, *stream);
}

CVAPI(void) cuda_imgproc_gammaCorrection(cv::_InputArray* src, cv::_OutputArray* dst, bool forward, Stream* stream) {
	cv::cuda::gammaCorrection(*src, *dst, forward, *stream);
}

CVAPI(void) cuda_imgproc_calcHist_0(cv::_InputArray* src, cv::_OutputArray* hist, Stream* stream) {
	cv::cuda::calcHist(*src, *hist, *stream);
}

CVAPI(void) cuda_imgproc_calcHist_1(cv::_InputArray* src, cv::_InputArray* mask,cv::_OutputArray* hist, Stream* stream) {
	cv::cuda::calcHist(*src, *mask, *hist, *stream);
}

CVAPI(void) cuda_imgproc_equalizeHist(cv::_InputArray* src, cv::_OutputArray* dst, Stream* stream) {
	cv::cuda::equalizeHist(*src, *dst, *stream);
}

// CLAHE
CVAPI(cv::Ptr<cv::cuda::CLAHE>*) cuda_imgproc_createCLAHE(double clipLimit, MyCvSize tileGridSize)
{
	cv::Ptr<cv::cuda::CLAHE> ret = cv::cuda::createCLAHE(clipLimit, cpp(tileGridSize));
	return clone(ret);
}

CVAPI(void) cuda_imgproc_Ptr_CLAHE_delete(cv::Ptr<cv::cuda::CLAHE>* obj)
{
	delete obj;
}

CVAPI(cv::cuda::CLAHE*) cuda_imgproc_Ptr_CLAHE_get(cv::Ptr<cv::cuda::CLAHE>* obj)
{
	return obj->get();
}


CVAPI(void) cuda_imgproc_CLAHE_apply(cv::cuda::CLAHE* obj, cv::_InputArray* src, cv::_OutputArray* dst, Stream* stream)
{
	obj->apply(*src, *dst, *stream);
}

CVAPI(void) cuda_imgproc_CLAHE_collectGarbage(cv::cuda::CLAHE* obj)
{
	obj->collectGarbage();
}
//

CVAPI(void) cuda_imgproc_evenLevels(cv::_OutputArray* levels, int nLevels, int lowerLevel, int upperLevel, Stream* stream) {
	cv::cuda::evenLevels(*levels, nLevels, lowerLevel, upperLevel, *stream);
}

CVAPI(void) cuda_imgproc_histEven_0(cv::_InputArray* src, cv::_OutputArray* hist, int histSize, int lowerLevel, int upperLevel, Stream* stream) {
	cv::cuda::histEven(*src, *hist, histSize, lowerLevel, upperLevel, *stream);
}

CVAPI(void) cuda_imgproc_histRange_0(cv::_InputArray* src, cv::_OutputArray* hist, cv::_InputArray* levels, Stream* stream) {
	cv::cuda::histRange(*src, *hist, *levels, *stream);
}

// Canny
CVAPI(cv::Ptr<CannyEdgeDetector>*) cuda_imgproc_createCannyEdgeDetector(double low_thresh, double high_thresh, int apperture_size, bool L2gradient)
{
	cv::Ptr<CannyEdgeDetector> ptr = cv::cuda::createCannyEdgeDetector(low_thresh, high_thresh, apperture_size, L2gradient);
	return new cv::Ptr<CannyEdgeDetector>(ptr);
}

CVAPI(void) cuda_imgproc_CannyEdgeDetector_detect(CannyEdgeDetector *obj, cv::_InputArray *image, cv::_OutputArray *edges, Stream* stream)
{
	obj->detect(*image, *edges, *stream);
}

CVAPI(void) cuda_imgproc_Ptr_CannyEdgeDetector_delete(cv::Ptr<CannyEdgeDetector> *obj)
{
	delete obj;
}

CVAPI(CannyEdgeDetector*) cuda_imgproc_Ptr_CannyEdgeDetector_get(cv::Ptr<CannyEdgeDetector> *ptr)
{
	return ptr->get();
}

CVAPI(double) cuda_imgproc_CannyEdgeDetector_getLowThreshold(CannyEdgeDetector* ptr)
{
	return ptr->getLowThreshold();
}
CVAPI(void) cuda_imgproc_CannyEdgeDetector_setLowThreshold(CannyEdgeDetector* ptr, double low_thresh)
{
	ptr->setLowThreshold(low_thresh);
}

CVAPI(double) cuda_imgproc_CannyEdgeDetector_getHighThreshold(CannyEdgeDetector* ptr)
{
	return ptr->getHighThreshold();
}
CVAPI(void) cuda_imgproc_CannyEdgeDetector_setHighThreshold(CannyEdgeDetector* ptr, double high_thresh)
{
	ptr->setHighThreshold(high_thresh);
}

CVAPI(int) cuda_imgproc_CannyEdgeDetector_getAppertureSize(CannyEdgeDetector* ptr)
{
	return ptr->getAppertureSize();
}
CVAPI(void) cuda_imgproc_CannyEdgeDetector_setAppertureSize(CannyEdgeDetector* ptr, int apperture_size)
{
	ptr->setAppertureSize(apperture_size);
}

CVAPI(bool) cuda_imgproc_CannyEdgeDetector_getL2Gradient(CannyEdgeDetector* ptr)
{
	return ptr->getL2Gradient();
}
CVAPI(void) cuda_imgproc_CannyEdgeDetector_setL2Gradient(CannyEdgeDetector* ptr, bool L2gradient)
{
	ptr->setL2Gradient(L2gradient);
}
//

// HoughLines
CVAPI(cv::Ptr<HoughLinesDetector>*) cuda_imgproc_createHoughLinesDetector(float rho, float theta, int threshold, bool doSort, int maxLines)
{
	cv::Ptr<HoughLinesDetector> ptr = cv::cuda::createHoughLinesDetector(rho, theta, threshold, doSort, maxLines);
	return new cv::Ptr<HoughLinesDetector>(ptr);
}

CVAPI(void) cuda_imgproc_HoughLinesDetector_detect(HoughLinesDetector* obj, cv::_InputArray* src, cv::_OutputArray* lines, Stream* stream)
{
	obj->detect(*src, *lines, *stream);
}

CVAPI(void) cuda_imgproc_HoughLinesDetector_downloadResults(HoughLinesDetector* obj, cv::_InputArray* d_lines, std::vector<cv::Vec2f>* h_lines, cv::_OutputArray* h_votes, Stream* stream)
{
	if (h_votes == nullptr)
		obj->downloadResults(*d_lines, *h_lines, cv::noArray(), *stream);
	else
		obj->downloadResults(*d_lines, *h_lines, *h_votes, *stream);
}

CVAPI(void) cuda_imgproc_Ptr_HoughLinesDetector_delete(cv::Ptr<HoughLinesDetector>* obj)
{
	delete obj;
}

CVAPI(HoughLinesDetector*) cuda_imgproc_Ptr_HoughLinesDetector_get(cv::Ptr<HoughLinesDetector>* ptr)
{
	return ptr->get();
}

CVAPI(float) cuda_imgproc_HoughLinesDetector_getRho(HoughLinesDetector* ptr)
{
	return ptr->getRho();
}
CVAPI(void) cuda_imgproc_HoughLinesDetector_setRho(HoughLinesDetector* ptr, float rho)
{
	ptr->setRho(rho);
}

CVAPI(float) cuda_imgproc_HoughLinesDetector_getTheta(HoughLinesDetector* ptr)
{
	return ptr->getTheta();
}
CVAPI(void) cuda_imgproc_HoughLinesDetector_setTheta(HoughLinesDetector* ptr, float theta)
{
	ptr->setTheta(theta);
}

CVAPI(int) cuda_imgproc_HoughLinesDetector_getThreshold(HoughLinesDetector* ptr)
{
	return ptr->getThreshold();
}
CVAPI(void) cuda_imgproc_HoughLinesDetector_setThreshold(HoughLinesDetector* ptr, int threshold)
{
	ptr->setThreshold(threshold);
}

CVAPI(bool) cuda_imgproc_HoughLinesDetector_getDoSort(HoughLinesDetector* ptr)
{
	return ptr->getDoSort();
}
CVAPI(void) cuda_imgproc_HoughLinesDetector_setDoSort(HoughLinesDetector* ptr, bool doSort)
{
	ptr->setDoSort(doSort);
}

CVAPI(int) cuda_imgproc_HoughLinesDetector_getMaxLines(HoughLinesDetector* ptr)
{
	return ptr->getMaxLines();
}
CVAPI(void) cuda_imgproc_HoughLinesDetector_setMaxLines(HoughLinesDetector* ptr, int maxLines)
{
	ptr->setMaxLines(maxLines);
}
//

// HoughSegmentDetector
CVAPI(cv::Ptr<HoughSegmentDetector>*) cuda_imgproc_createHoughSegmentDetector(float rho, float theta, int minLineLength, int maxLineGap, int maxLines)
{
	cv::Ptr<HoughSegmentDetector> ptr = cv::cuda::createHoughSegmentDetector(rho, theta, minLineLength, maxLineGap, maxLines);
	return new cv::Ptr<HoughSegmentDetector>(ptr);
}

CVAPI(void) cuda_imgproc_HoughSegmentDetector_detect(HoughSegmentDetector* obj, cv::_InputArray* src, cv::_OutputArray* lines, Stream* stream)
{
	obj->detect(*src, *lines, *stream);
}

CVAPI(void) cuda_imgproc_HoughSegmentDetector_downloadResults(HoughSegmentDetector* obj, cv::_InputArray* _d_lines, std::vector<cv::Vec4i>* h_lines, Stream* stream)
{
	GpuMat d_lines = _d_lines->getGpuMat();
	if (*stream)
		d_lines.row(0).download(*h_lines, *stream);
	else
		d_lines.row(0).download(*h_lines);
}

CVAPI(void) cuda_imgproc_Ptr_HoughSegmentDetector_delete(cv::Ptr<HoughSegmentDetector>* obj)
{
	delete obj;
}

CVAPI(HoughSegmentDetector*) cuda_imgproc_Ptr_HoughSegmentDetector_get(cv::Ptr<HoughSegmentDetector>* ptr)
{
	return ptr->get();
}

CVAPI(float) cuda_imgproc_HoughSegmentDetector_getRho(HoughSegmentDetector* ptr)
{
	return ptr->getRho();
}
CVAPI(void) cuda_imgproc_HoughSegmentDetector_setRho(HoughSegmentDetector* ptr, float rho)
{
	ptr->setRho(rho);
}

CVAPI(float) cuda_imgproc_HoughSegmentDetector_getTheta(HoughLinesDetector* ptr)
{
	return ptr->getTheta();
}
CVAPI(void) cuda_imgproc_HoughSegmentDetector_setTheta(HoughLinesDetector* ptr, float theta)
{
	ptr->setTheta(theta);
}

CVAPI(int) cuda_imgproc_HoughSegmentDetector_getMinLineLength(HoughSegmentDetector* ptr)
{
	return ptr->getMinLineLength();
}
CVAPI(void) cuda_imgproc_HoughSegmentDetector_setMinLineLength(HoughSegmentDetector* ptr, int minLineLength)
{
	ptr->setMinLineLength(minLineLength);
}

CVAPI(int) cuda_imgproc_HoughSegmentDetector_getMaxLineGap(HoughSegmentDetector* ptr)
{
	return ptr->getMaxLines();
}
CVAPI(void) cuda_imgproc_HoughSegmentDetector_setMaxLineGap(HoughSegmentDetector* ptr, int maxLineGap)
{
	ptr->setMaxLineGap(maxLineGap);
}

CVAPI(int) cuda_imgproc_HoughSegmentDetector_getMaxLines(HoughSegmentDetector* ptr)
{
	return ptr->getMaxLines();
}
CVAPI(void) cuda_imgproc_HoughSegmentDetector_setMaxLines(HoughSegmentDetector* ptr, int maxLines)
{
	ptr->setMaxLines(maxLines);
}
//

// HoughCirclesDetector
CVAPI(cv::Ptr<HoughCirclesDetector>*) cuda_imgproc_createHoughCirclesDetector(float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles)
{
	cv::Ptr<HoughCirclesDetector> ptr = cv::cuda::createHoughCirclesDetector(dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius, maxCircles);
	return new cv::Ptr<HoughCirclesDetector>(ptr);
}

CVAPI(void) cuda_imgproc_HoughCirclesDetector_detect(HoughCirclesDetector* obj, cv::_InputArray* src, cv::_OutputArray* circles, Stream* stream)
{
	obj->detect(*src, *circles, *stream);
}

CVAPI(void) cuda_imgproc_HoughCirclesDetector_downloadResults(HoughCirclesDetector* obj, cv::_InputArray* _d_circles, std::vector<cv::Vec3f>* c_circles, Stream* stream)
{
	GpuMat d_circles = _d_circles->getGpuMat();
	if (*stream)
		d_circles.download(*c_circles, *stream);
	else
		d_circles.download(*c_circles);
}

CVAPI(void) cuda_imgproc_Ptr_HoughCirclesDetector_delete(cv::Ptr<HoughCirclesDetector>* obj)
{
	delete obj;
}

CVAPI(HoughCirclesDetector*) cuda_imgproc_Ptr_HoughCirclesDetector_get(cv::Ptr<HoughCirclesDetector>* ptr)
{
	return ptr->get();
}

CVAPI(float) cuda_imgproc_HoughCirclesDetector_getDp(HoughCirclesDetector* ptr)
{
	return ptr->getDp();
}
CVAPI(void) cuda_imgproc_HoughCirclesDetector_setDp(HoughCirclesDetector* ptr, float dp)
{
	ptr->setDp(dp);
}

CVAPI(float) cuda_imgproc_HoughCirclesDetector_getMinDist(HoughCirclesDetector* ptr)
{
	return ptr->getMinDist();
}
CVAPI(void) cuda_imgproc_HoughCirclesDetector_setMinDist(HoughCirclesDetector* ptr, float minDist)
{
	ptr->setMinDist(minDist);
}

CVAPI(int) cuda_imgproc_HoughCirclesDetector_getCannyThreshold(HoughCirclesDetector* ptr)
{
	return ptr->getCannyThreshold();
}
CVAPI(void) cuda_imgproc_HoughCirclesDetector_setCannyThreshold(HoughCirclesDetector* ptr, int cannyThreshold)
{
	ptr->setCannyThreshold(cannyThreshold);
}

CVAPI(int) cuda_imgproc_HoughCirclesDetector_getVotesThreshold(HoughCirclesDetector* ptr)
{
	return ptr->getVotesThreshold();
}
CVAPI(void) cuda_imgproc_HoughCirclesDetector_setVotesThreshold(HoughCirclesDetector* ptr, int votesThreshold)
{
	ptr->setVotesThreshold(votesThreshold);
}

CVAPI(int) cuda_imgproc_HoughCirclesDetector_getMinRadius(HoughCirclesDetector* ptr)
{
	return ptr->getMinRadius();
}
CVAPI(void) cuda_imgproc_HoughCirclesDetector_setMinRadius(HoughCirclesDetector* ptr, int minRadius)
{
	ptr->setMinRadius(minRadius);
}

CVAPI(int) cuda_imgproc_HoughCirclesDetector_getMaxRadius(HoughCirclesDetector* ptr)
{
	return ptr->getMaxRadius();
}
CVAPI(void) cuda_imgproc_HoughCirclesDetector_setMaxRadius(HoughCirclesDetector* ptr, int maxRadius)
{
	ptr->setMaxRadius(maxRadius);
}

CVAPI(int) cuda_imgproc_HoughCirclesDetector_getMaxCircles(HoughCirclesDetector* ptr)
{
	return ptr->getMaxCircles();
}
CVAPI(void) cuda_imgproc_HoughCirclesDetector_setMaxCircles(HoughCirclesDetector* ptr, int maxCircles)
{
	ptr->setMaxCircles(maxCircles);
}


#endif

#endif