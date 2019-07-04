#if ENABLED_CUDA

using System;
using System.Runtime.InteropServices;

#pragma warning disable 1591

namespace OpenCvSharp {
    // ReSharper disable InconsistentNaming

    public static partial class NativeMethods {
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_demosaicing(IntPtr src, IntPtr dst, int code, int dcn, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_swapChannels(IntPtr src, int[] dstOrder, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_gammaCorrection(IntPtr src, IntPtr dst, bool forward, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_calcHist_0(IntPtr src, IntPtr hist, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_calcHist_1(IntPtr src, IntPtr mask, IntPtr hist, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_equalizeHist(IntPtr src, IntPtr dst, IntPtr stream);

        //CLAHE
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr cuda_imgproc_createCLAHE(double clipLimit, Size tileGridSize);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_Ptr_CLAHE_delete(IntPtr obj);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr cuda_imgproc_Ptr_CLAHE_get(IntPtr obj);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr cuda_imgproc_CLAHE_collectGarbage(IntPtr obj);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr cuda_imgproc_CLAHE_apply(IntPtr obj, IntPtr src, IntPtr dst, IntPtr stream);
        //

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_evenLevels(IntPtr levels, int nLevels, int lowerLevel, int upperLevel, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_histEven_0(IntPtr src, IntPtr hist, int histSize, int lowerLevel, int upperLevel, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_histRange_0(IntPtr src, IntPtr hist, IntPtr levels, IntPtr stream);

        // Cany
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr cuda_imgproc_createCannyEdgeDetector(double low_thresh, double high_thresh, int apperture_size, bool L2gradient);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_CannyEdgeDetector_detect(IntPtr self, IntPtr image, IntPtr edges, IntPtr stream);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_Ptr_CannyEdgeDetector_delete(IntPtr obj);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr cuda_imgproc_Ptr_CannyEdgeDetector_get(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern double cuda_imgproc_CannyEdgeDetector_getLowThreshold(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_CannyEdgeDetector_setLowThreshold(IntPtr ptr, double low_thresh);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern double cuda_imgproc_CannyEdgeDetector_getHighThreshold(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_CannyEdgeDetector_setHighThreshold(IntPtr ptr, double high_thresh);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_imgproc_CannyEdgeDetector_getAppertureSize(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_CannyEdgeDetector_setAppertureSize(IntPtr ptr, int apperture_size);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern bool cuda_imgproc_CannyEdgeDetector_getL2Gradient(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_CannyEdgeDetector_setL2Gradient(IntPtr ptr, bool L2gradient);
        //

        // HoughLines
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr cuda_imgproc_createHoughLinesDetector(float rho, float theta, int threshold, bool doSort, int maxLines);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughLinesDetector_detect(IntPtr self ,IntPtr src, IntPtr lines, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughLinesDetector_downloadResults(IntPtr self ,IntPtr d_lines, IntPtr h_lines, IntPtr h_votes, IntPtr stream);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_Ptr_HoughLinesDetector_delete(IntPtr obj);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr cuda_imgproc_Ptr_HoughLinesDetector_get(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern float cuda_imgproc_HoughLinesDetector_getRho(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughLinesDetector_setRho(IntPtr ptr, float rho);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern float cuda_imgproc_HoughLinesDetector_getTheta(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughLinesDetector_setTheta(IntPtr ptr, float theta);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_imgproc_HoughLinesDetector_getThreshold(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughLinesDetector_setThreshold(IntPtr ptr, int threshold);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern bool cuda_imgproc_HoughLinesDetector_getDoSort(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughLinesDetector_setDoSort(IntPtr ptr, bool doSort);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_imgproc_HoughLinesDetector_getMaxLines(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughLinesDetector_setMaxLines(IntPtr ptr, int maxLines);
        //

        // HoughSegmentDetector
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr cuda_imgproc_createHoughSegmentDetector(float rho, float theta, int minLineLength, int maxLineGap, int maxLines);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughSegmentDetector_detect(IntPtr self, IntPtr src, IntPtr lines, IntPtr stream);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_Ptr_HoughSegmentDetector_delete(IntPtr obj);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughSegmentDetector_downloadResults(IntPtr obj, IntPtr d_lines, IntPtr h_lines, IntPtr stream);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr cuda_imgproc_Ptr_HoughSegmentDetector_get(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern float cuda_imgproc_HoughSegmentDetector_getRho(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughSegmentDetector_setRho(IntPtr ptr, float rho);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern float cuda_imgproc_HoughSegmentDetector_getTheta(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughSegmentDetector_setTheta(IntPtr ptr, float theta);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_imgproc_HoughSegmentDetector_getMinLineLength(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughSegmentDetector_setMinLineLength(IntPtr ptr, int minLineLength);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_imgproc_HoughSegmentDetector_getMaxLineGap(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughSegmentDetector_setMaxLineGap(IntPtr ptr, int maxLineGap);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_imgproc_HoughSegmentDetector_getMaxLines(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughSegmentDetector_setMaxLines(IntPtr ptr, int maxLines);
        //

        //HoughCircles
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr cuda_imgproc_createHoughCirclesDetector(float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughCirclesDetector_detect(IntPtr self, IntPtr src, IntPtr circles, IntPtr stream);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_Ptr_HoughCirclesDetector_delete(IntPtr obj);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughCirclesDetector_downloadResults(IntPtr obj, IntPtr d_circles, IntPtr c_circles, IntPtr stream);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr cuda_imgproc_Ptr_HoughCirclesDetector_get(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern float cuda_imgproc_HoughCirclesDetector_getDp(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughCirclesDetector_setDp(IntPtr ptr, float dp);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern float cuda_imgproc_HoughCirclesDetector_getMinDist(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughCirclesDetector_setMinDist(IntPtr ptr, float minDist);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_imgproc_HoughCirclesDetector_getCannyThreshold(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughCirclesDetector_setCannyThreshold(IntPtr ptr, int cannyThreshold);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_imgproc_HoughCirclesDetector_getVotesThreshold(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughCirclesDetector_setVotesThreshold(IntPtr ptr, int votesThreshold);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_imgproc_HoughCirclesDetector_getMinRadius(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughCirclesDetector_setMinRadius(IntPtr ptr, int minRadius);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_imgproc_HoughCirclesDetector_getMaxRadius(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughCirclesDetector_setMaxRadius(IntPtr ptr, int maxRadius);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_imgproc_HoughCirclesDetector_getMaxCircles(IntPtr ptr);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_HoughCirclesDetector_setMaxCircles(IntPtr ptr, int maxCircles);

    }

}
#endif