using System;

namespace OpenCvSharp.Cuda {
    // ReSharper disable InconsistentNaming

    /// <summary>
    /// Base class for Contrast Limited Adaptive Histogram Equalization. :
    /// </summary>
    public class CLAHE : Algorithm {
        /// <summary>
        /// cv::Ptr&lt;T&gt;
        /// </summary>
        private Ptr ptrObj;

        /// <summary>
        /// 
        /// </summary>
        private CLAHE(IntPtr p) {
            ptrObj = new Ptr(p);
            ptr = ptrObj.Get();
        }

        #region Init & Disposal

        /// <summary>
        /// Creates implementation for cuda::CLAHE .
        /// </summary>
        /// <param name="clipLimit">Threshold for contrast limiting.</param>
        /// <param name="tileGridSize">Size of grid for histogram equalization. Input image will be divided into
        /// equally sized rectangular tiles.tileGridSize defines the number of tiles in row and column.</param>
        /// <returns></returns>
        public static CLAHE create(double clipLimit = 40.0, Size? tileGridSize = null) {
            IntPtr ptr = NativeMethods.cuda_imgproc_createCLAHE(clipLimit, tileGridSize.GetValueOrDefault(new Size(8, 8)));
            return new CLAHE(ptr);
        }

        /// <summary>
        /// Releases managed resources
        /// </summary>
        protected override void DisposeManaged() {
            ptrObj?.Dispose();
            ptrObj = null;
            base.DisposeManaged();
        }
        #endregion

        /// <summary>
        /// 
        /// </summary>
        public void CollectGarbage() {
            ThrowIfDisposed();
            NativeMethods.cuda_imgproc_CLAHE_collectGarbage(ptr);
            GC.KeepAlive(this);
        }

        /// <summary>
        /// Equalizes the histogram of a grayscale image using Contrast Limited Adaptive Histogram Equalization.
        /// </summary>
        /// <param name="src">Source image with CV_8UC1 type.</param>
        /// <param name="dst">Destination image.</param>
        /// <param name="stream">stream Stream for the asynchronous version.</param>
        public void Apply(InputArray src, OutputArray dst, Stream stream = null) {
            ThrowIfDisposed();
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.cuda_imgproc_CLAHE_apply(ptr, src.CvPtr, dst.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);

            dst.Fix();
            GC.KeepAlive(this);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
        }

        internal class Ptr : OpenCvSharp.Ptr {
            public Ptr(IntPtr ptr) : base(ptr) {
            }

            public override IntPtr Get() {
                var res = NativeMethods.cuda_imgproc_Ptr_CLAHE_get(ptr);
                GC.KeepAlive(this);
                return res;
            }

            protected override void DisposeUnmanaged() {
                NativeMethods.cuda_imgproc_Ptr_CLAHE_delete(ptr);
                base.DisposeUnmanaged();
            }
        }
    }
}