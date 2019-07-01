#if ENABLED_CUDA

using System;
using System.Runtime.InteropServices;

#pragma warning disable 1591

namespace OpenCvSharp {
    // ReSharper disable InconsistentNaming

    public static partial class NativeMethods {

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_warping_remap(IntPtr src, IntPtr dst, IntPtr map1, IntPtr map2, int interpolation, int borderMode, Scalar borderValue, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_warping_warpAffine(IntPtr src, IntPtr dst, IntPtr m, Size dsize, int flags, int borderMode, Scalar borderValue, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_warping_buildWarpAffineMaps(IntPtr M, bool inverse, Size dsize, IntPtr xmap, IntPtr ymap, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_warping_warpPerspective(IntPtr src, IntPtr dst, IntPtr m, Size dsize, int flags, int borderMode, Scalar borderValue, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_warping_buildWarpPerspectiveMaps(IntPtr M, bool inverse, Size dsize, IntPtr xmap, IntPtr ymap, IntPtr stream);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_warping_rotate(IntPtr src, IntPtr dst, Size dsize, double angle, double xShift, double yShift, int flags, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_warping_resize(IntPtr src, IntPtr dst, Size dsize, double fx, double fy, int interpolation, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_warping_pyrDown(IntPtr src, IntPtr dst, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void cuda_warping_pyrUp(IntPtr src, IntPtr dst, IntPtr stream);

    }

}
#endif