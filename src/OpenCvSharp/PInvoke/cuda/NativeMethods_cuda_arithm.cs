#if ENABLED_CUDA

using System;
using System.Runtime.InteropServices;

#pragma warning disable 1591

namespace OpenCvSharp {
    // ReSharper disable InconsistentNaming

    public static partial class NativeMethods {

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int cuda_countNonZero(IntPtr src);

    }

}
#endif
