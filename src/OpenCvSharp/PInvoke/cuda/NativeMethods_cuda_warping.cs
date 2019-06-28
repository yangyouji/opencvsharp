#if ENABLED_CUDA

using System;
using System.Runtime.InteropServices;

#pragma warning disable 1591

namespace OpenCvSharp {
    // ReSharper disable InconsistentNaming

    public static partial class NativeMethods {

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_resize(IntPtr src, IntPtr dst, Size dsize, double fx, double fy, int interpolation, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_pyrDown(IntPtr src, IntPtr dst, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_imgproc_pyrUp(IntPtr src, IntPtr dst, IntPtr stream);

        //[DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        //public static extern void cuda_imgproc_pyrUp(IntPtr src, IntPtr dst, IntPtr stream);

    }

}
#endif