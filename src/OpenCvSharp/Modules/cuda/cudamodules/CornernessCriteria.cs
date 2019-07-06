using System;

namespace OpenCvSharp.Cuda {
    // ReSharper disable InconsistentNaming

    /// <summary>
    /// Computes the cornerness criteria at each image pixel.
    /// </summary>
    public class CornernessCriteria : Algorithm {
        /// <summary>
        /// cv::Ptr&lt;T&gt;
        /// </summary>
        private Ptr objectPtr;

        #region Init & Disposal

        /// <summary>
        /// Creates implementation for Harris cornerness criteria.
        /// </summary>
        /// <param name="srcType">Input source type. Only CV_8UC1 and CV_32FC1 are supported for now.</param>
        /// <param name="blockSize">Neighborhood size.</param>
        /// <param name="ksize">Aperture parameter for the Sobel operator.</param>
        /// <param name="k">Harris detector free parameter.</param>
        /// <param name="borderType">Pixel extrapolation method. Only BORDER_REFLECT101 and BORDER_REPLICATE are
        /// supported for now.</param>
        /// <returns></returns>
        public static CornernessCriteria createHarrisCorner(int srcType, int blockSize, int ksize, double k, BorderTypes borderType = BorderTypes.Reflect101) {
            IntPtr ptr = NativeMethods.cuda_imgproc_createHarrisCorner(
                srcType, blockSize, ksize, k, (int)borderType);
            return new CornernessCriteria(ptr);
        }

        /// <summary>
        /// Creates implementation for the minimum eigen value of a 2x2 derivative covariation matrix (the
        /// cornerness criteria).
        /// </summary>
        /// <param name="srcType">Input source type. Only CV_8UC1 and CV_32FC1 are supported for now.</param>
        /// <param name="blockSize">Neighborhood size.</param>
        /// <param name="ksize">Aperture parameter for the Sobel operator.</param>
        /// <param name="borderType">Pixel extrapolation method. Only BORDER_REFLECT101 and BORDER_REPLICATE are
        /// supported for now.</param>
        /// <returns></returns>
        public static CornernessCriteria createMinEigenValCorner(int srcType, int blockSize, int ksize, BorderTypes borderType = BorderTypes.Reflect101) {
            IntPtr ptr = NativeMethods.cuda_imgproc_createMinEigenValCorner(
                srcType, blockSize, ksize, (int)borderType);
            return new CornernessCriteria(ptr);
        }

        internal CornernessCriteria(IntPtr ptr) {
            this.objectPtr = new Ptr(ptr);
            this.ptr = objectPtr.Get();
        }

        /// <summary>
        /// Releases managed resources
        /// </summary>
        protected override void DisposeManaged() {
            objectPtr?.Dispose();
            objectPtr = null;
            base.DisposeManaged();
        }

        #endregion

        /// <summary>
        /// Computes the cornerness criteria at each image pixel.
        /// </summary>
        /// <param name="src">Source image.</param>
        /// <param name="dst">Destination image containing cornerness values. It will have the same size as src and
        /// CV_32FC1 type.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public virtual void compute(InputArray src, OutputArray dst, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.cuda_imgproc_HoughCirclesDetector_compute(ptr, src.CvPtr, dst.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);

            dst.Fix();
            GC.KeepAlive(this);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
        }

        internal class Ptr : OpenCvSharp.Ptr {
            public Ptr(IntPtr ptr) : base(ptr) {
            }

            public override IntPtr Get() {
                var res = NativeMethods.cuda_imgproc_Ptr_CornernessCriteria_get(ptr);
                GC.KeepAlive(this);
                return res;
            }

            protected override void DisposeUnmanaged() {
                NativeMethods.cuda_imgproc_Ptr_CornernessCriteria_delete(ptr);
                base.DisposeUnmanaged();
            }
        }
    }
}