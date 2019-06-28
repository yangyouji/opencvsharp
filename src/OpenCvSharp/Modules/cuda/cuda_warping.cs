using System;
using System.Collections.Generic;
using System.Text;

namespace OpenCvSharp.Cuda {

    /// <summary>
    ///  GPU warping
    /// </summary>
    public static partial class cuda {
        /// <summary>
        /// GPU Resize
        /// </summary>
        public static void Resize(InputArray src, OutputArray dst, Size dsize,
            double fx = 0, double fy = 0, InterpolationFlags interpolation = InterpolationFlags.Linear, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.cuda_imgproc_resize(src.CvPtr, dst.CvPtr, dsize, fx, fy, (int)interpolation, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
        }

        /// <summary>
        /// GPU pyrDown
        /// </summary>
        public static void pyrDown(InputArray src, OutputArray dst, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.cuda_imgproc_pyrDown(src.CvPtr, dst.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
        }

        /// <summary>
        /// GPU pyrUp
        /// </summary>
        public static void pyrUp(InputArray src, OutputArray dst, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.cuda_imgproc_pyrUp(src.CvPtr, dst.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
        }

    }
}
