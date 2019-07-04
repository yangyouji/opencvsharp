using System;

namespace OpenCvSharp.Cuda {
    // ReSharper disable InconsistentNaming

    /// <summary>
    /// Base class for Canny Edge Detector. :
    /// </summary>
    public class CannyEdgeDetector : Algorithm {
        /// <summary>
        /// cv::Ptr&lt;T&gt;
        /// </summary>
        private Ptr objectPtr;

        #region Init & Disposal

        /// <summary>
        /// 
        /// </summary>
        /// <param name="low_thresh"></param>
        /// <param name="high_thresh"></param>
        /// <param name="apperture_size"></param>
        /// <param name="L2gradient"></param>
        /// <returns></returns>
        public static CannyEdgeDetector create(
            double low_thresh, double high_thresh, int apperture_size = 3, bool L2gradient = false) {
            IntPtr ptr = NativeMethods.cuda_imgproc_createCannyEdgeDetector(
                low_thresh, high_thresh, apperture_size, L2gradient);
            return new CannyEdgeDetector(ptr);
        }

        internal CannyEdgeDetector(IntPtr ptr) {
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
        /// Finds edges in an image using the @cite Canny86 algorithm.
        /// </summary>
        /// <param name="image">Single-channel 8-bit input image.</param>
        /// <param name="edges">Output edge map. It has the same size and type as image.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public virtual void detect(InputArray image, OutputArray edges, Stream stream = null) {
            if (image == null)
                throw new ArgumentNullException(nameof(image));
            if (edges == null)
                throw new ArgumentNullException(nameof(edges));
            image.ThrowIfDisposed();
            edges.ThrowIfNotReady();

            NativeMethods.cuda_imgproc_CannyEdgeDetector_detect(ptr, image.CvPtr, edges.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);

            edges.Fix();
            GC.KeepAlive(this);
            GC.KeepAlive(image);
            GC.KeepAlive(edges);
        }


        #region Properties

        /// <summary>
        /// 
        /// </summary>
        public double LowThreshold {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_CannyEdgeDetector_getLowThreshold(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_CannyEdgeDetector_setLowThreshold(ptr, value);
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public double HighThreshold {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_CannyEdgeDetector_getHighThreshold(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_CannyEdgeDetector_setHighThreshold(ptr, value);
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public int AppertureSize {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_CannyEdgeDetector_getAppertureSize(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_CannyEdgeDetector_setAppertureSize(ptr, value);
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public bool L2Gradient {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_CannyEdgeDetector_getL2Gradient(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_CannyEdgeDetector_setL2Gradient(ptr, value);
                GC.KeepAlive(this);
            }
        }

        #endregion

        internal class Ptr : OpenCvSharp.Ptr {
            public Ptr(IntPtr ptr) : base(ptr) {
            }

            public override IntPtr Get() {
                var res = NativeMethods.cuda_imgproc_Ptr_CannyEdgeDetector_get(ptr);
                GC.KeepAlive(this);
                return res;
            }

            protected override void DisposeUnmanaged() {
                NativeMethods.cuda_imgproc_Ptr_CannyEdgeDetector_delete(ptr);
                base.DisposeUnmanaged();
            }
        }
    }
}