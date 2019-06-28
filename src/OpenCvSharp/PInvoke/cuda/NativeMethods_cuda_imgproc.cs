#if ENABLED_CUDA

using System;
using System.Runtime.InteropServices;

#pragma warning disable 1591

namespace OpenCvSharp {
    // ReSharper disable InconsistentNaming

    public static partial class NativeMethods {

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr cuda_createCannyEdgeDetector(double low_thresh, double high_thresh, int apperture_size = 3, bool L2gradient = false);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_CannyEdgeDetector_detect(IntPtr self, IntPtr image, IntPtr edges, IntPtr stream);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_Ptr_CannyEdgeDetector_delete(IntPtr obj);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern IntPtr cuda_Ptr_CannyEdgeDetector_get(IntPtr ptr);

    }

}
#endif