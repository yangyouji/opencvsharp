using System;

namespace OpenCvSharp.Cuda {
    // ReSharper disable InconsistentNaming

    /// <summary>
    /// Finds circles in a grayscale image using the Hough transform.
    /// </summary>
    public class HoughCirclesDetector : Algorithm {
        /// <summary>
        /// cv::Ptr&lt;T&gt;
        /// </summary>
        private Ptr objectPtr;

        #region Init & Disposal

        /// <summary>
        /// Creates implementation for cuda::HoughCirclesDetector .
        /// </summary>
        /// <param name="dp">Inverse ratio of the accumulator resolution to the image resolution. For example, if
        /// dp=1 , the accumulator has the same resolution as the input image.If dp = 2, the accumulator has
        /// half as big width and height.</param>
        /// <param name="minDist">Minimum distance between the centers of the detected circles. If the parameter is
        /// too small, multiple neighbor circles may be falsely detected in addition to a true one.If it is
        /// too large, some circles may be missed.</param>
        /// <param name="cannyThreshold">The higher threshold of the two passed to Canny edge detector (the lower one
        /// is twice smaller).</param>
        /// <param name="votesThreshold">The accumulator threshold for the circle centers at the detection stage. The
        /// smaller it is, the more false circles may be detected.</param>
        /// <param name="minRadius">Minimum circle radius.</param>
        /// <param name="maxRadius">Maximum circle radius.</param>
        /// <param name="maxCircles">Maximum number of output circles.</param>
        /// <returns></returns>
        public static HoughCirclesDetector create(float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles = 4096) {
            IntPtr ptr = NativeMethods.cuda_imgproc_createHoughCirclesDetector(
                dp, minDist,cannyThreshold,votesThreshold,minRadius,maxRadius,maxCircles);
            return new HoughCirclesDetector(ptr);
        }

        internal HoughCirclesDetector(IntPtr ptr) {
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
        /// Finds circles in a grayscale image using the Hough transform.
        /// </summary>
        /// <param name="src">8-bit, single-channel grayscale input image.</param>
        /// <param name="circles">Output vector of found circles. Each vector is encoded as a 3-element
        /// floating-point vector \f$(x, y, radius)\f$ .</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public virtual void detect(InputArray src, OutputArray circles, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (circles == null)
                throw new ArgumentNullException(nameof(circles));
            src.ThrowIfDisposed();
            circles.ThrowIfNotReady();

            NativeMethods.cuda_imgproc_HoughCirclesDetector_detect(ptr, src.CvPtr, circles.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);

            circles.Fix();
            GC.KeepAlive(this);
            GC.KeepAlive(src);
            GC.KeepAlive(circles);
        }

        /// <summary>
        /// Downloads results from cuda::HoughCirclesDetector::detect to host memory.
        /// </summary>
        /// <param name="d_circles">Result of cuda::HoughCirclesDetector::detect .</param>
        /// <param name="c_circles">Output host array.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public virtual void downloadResults(InputArray d_circles, out CircleSegment[] c_circles, Stream stream = null) {
            if (d_circles == null)
                throw new ArgumentNullException(nameof(d_circles));
            d_circles.ThrowIfDisposed();

            using (var vec = new VectorOfVec3f()) {
                NativeMethods.cuda_imgproc_HoughCirclesDetector_downloadResults(ptr, d_circles.CvPtr, vec.CvPtr
                    , stream?.CvPtr ?? Stream.Null.CvPtr);
                c_circles = vec.ToArray<CircleSegment>();
            }

            GC.KeepAlive(this);
            GC.KeepAlive(d_circles);
            GC.KeepAlive(c_circles);
        }

        #region Properties

        /// <summary>
        /// 
        /// </summary>
        public float Dp {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_HoughCirclesDetector_getDp(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_HoughCirclesDetector_setDp(ptr, value);
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public float minDist {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_HoughCirclesDetector_getMinDist(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_HoughCirclesDetector_setMinDist(ptr, value);
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public int cannyThreshold {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_HoughCirclesDetector_getCannyThreshold(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_HoughCirclesDetector_setCannyThreshold(ptr, value);
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public int votesThreshold {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_HoughCirclesDetector_getVotesThreshold(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_HoughCirclesDetector_setVotesThreshold(ptr, value);
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public int minRadius {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_HoughCirclesDetector_getMinRadius(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_HoughCirclesDetector_setMinRadius(ptr, value);
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public int maxRadius {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_HoughCirclesDetector_getMaxRadius(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_HoughCirclesDetector_setMaxRadius(ptr, value);
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public int maxCircles {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_HoughCirclesDetector_getMaxCircles(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_HoughCirclesDetector_setMaxCircles(ptr, value);
                GC.KeepAlive(this);
            }
        }

        #endregion

        internal class Ptr : OpenCvSharp.Ptr {
            public Ptr(IntPtr ptr) : base(ptr) {
            }

            public override IntPtr Get() {
                var res = NativeMethods.cuda_imgproc_Ptr_HoughSegmentDetector_get(ptr);
                GC.KeepAlive(this);
                return res;
            }

            protected override void DisposeUnmanaged() {
                NativeMethods.cuda_imgproc_Ptr_HoughSegmentDetector_delete(ptr);
                base.DisposeUnmanaged();
            }
        }
    }
}