#if ENABLED_CUDA

using System;
using System.Runtime.InteropServices;

#pragma warning disable 1591

namespace OpenCvSharp {
    // ReSharper disable InconsistentNaming

    public static partial class NativeMethods {

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_add(IntPtr src1, IntPtr src2, IntPtr dst, IntPtr mask, int dtype, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_subtract(IntPtr src1, IntPtr src2, IntPtr dst, IntPtr mask, int dtype, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_multiply(IntPtr src1, IntPtr src2, IntPtr dst, double scale, int dtype, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_divide(IntPtr src1, IntPtr src2, IntPtr dst, double scale, int dtype, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_absdiff(IntPtr src1, IntPtr src2, IntPtr dst, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_abs(IntPtr src, IntPtr dst, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_sqr(IntPtr src, IntPtr dst, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_sqrt(IntPtr src, IntPtr dst, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_exp(IntPtr src, IntPtr dst, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_log(IntPtr src, IntPtr dst, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_pow(IntPtr src, double power, IntPtr dst, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_compare(IntPtr src1, IntPtr src2, IntPtr dst, int cmpop, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_bitwise_not(IntPtr src, IntPtr dst, IntPtr mask, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_bitwise_or(IntPtr src1, IntPtr src2, IntPtr dst, IntPtr mask, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_bitwise_and(IntPtr src1, IntPtr src2, IntPtr dst, IntPtr mask, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_bitwise_xor(IntPtr src1, IntPtr src2, IntPtr dst, IntPtr mask, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_min(IntPtr src1, IntPtr src2, IntPtr dst, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_max(IntPtr src1, IntPtr src2, IntPtr dst, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_addWeighted(IntPtr src1, double alpha, IntPtr src2, double beta, double gamma, IntPtr dst, int dtype, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_threshold(IntPtr src, IntPtr dst, double thresh, double maxval, int type, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_magnitude_0(IntPtr xy, IntPtr magnitude, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_magnitudeSqr_0(IntPtr xy, IntPtr magnitude, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_magnitude_1(IntPtr x, IntPtr y, IntPtr magnitude, IntPtr stream);
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_magnitudeSqr_1(IntPtr x, IntPtr y, IntPtr magnitude, IntPtr stream);




        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_arithm_countNonZero(IntPtr src);

    }

}
#endif
