using System;

namespace OpenCvSharp.Cuda {
    // ReSharper disable InconsistentNaming

    /// <summary>
    /// Base class for line segments detector algorithm. :
    /// </summary>
    public class HoughSegmentDetector : Algorithm {
        /// <summary>
        /// cv::Ptr&lt;T&gt;
        /// </summary>
        private Ptr objectPtr;

        #region Init & Disposal

        /// <summary>
        /// Creates implementation for cuda::HoughSegmentDetector .
        /// </summary>
        /// <param name="rho">Distance resolution of the accumulator in pixels.</param>
        /// <param name="theta">Angle resolution of the accumulator in radians.</param>
        /// <param name="minLineLength">Minimum line length. Line segments shorter than that are rejected.</param>
        /// <param name="maxLineGap">Maximum allowed gap between points on the same line to link them.</param>
        /// <param name="maxLines">Maximum number of output lines.</param>
        /// <returns></returns>
        public static HoughSegmentDetector create(float rho, float theta, int minLineLength, int maxLineGap, int maxLines = 4096) {
            IntPtr ptr = NativeMethods.cuda_imgproc_createHoughSegmentDetector(
                rho, theta, minLineLength, maxLineGap, maxLines);
            return new HoughSegmentDetector(ptr);
        }

        internal HoughSegmentDetector(IntPtr ptr) {
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
        /// Finds line segments in a binary image using the probabilistic Hough transform.
        /// </summary>
        /// <param name="src">8-bit, single-channel binary source image.</param>
        /// <param name="lines">Output vector of lines. Each line is represented by a 4-element vector
        /// \f$(x_1, y_1, x_2, y_2)\f$ , where \f$(x_1, y_1)\f$ and \f$(x_2, y_2)\f$ are the ending points of each detected
        /// line segment.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public virtual void detect(InputArray src, OutputArray lines, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (lines == null)
                throw new ArgumentNullException(nameof(lines));
            src.ThrowIfDisposed();
            lines.ThrowIfNotReady();

            NativeMethods.cuda_imgproc_HoughSegmentDetector_detect(ptr, src.CvPtr, lines.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);

            lines.Fix();
            GC.KeepAlive(this);
            GC.KeepAlive(src);
            GC.KeepAlive(lines);
        }

        /// <summary>
        /// Downloads results from cuda::HoughLinesDetector::detect to host memory.
        /// </summary>
        /// <param name="d_lines">Result of cuda::HoughLinesDetector::detect .</param>
        /// <param name="h_lines">Output host array.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public virtual void downloadResults(InputArray d_lines, out LineSegmentPoint[] h_lines, Stream stream = null) {
            if (d_lines == null)
                throw new ArgumentNullException(nameof(d_lines));
            d_lines.ThrowIfDisposed();

            using (var vec = new VectorOfVec4i()) {
                NativeMethods.cuda_imgproc_HoughSegmentDetector_downloadResults(ptr, d_lines.CvPtr, vec.CvPtr
                    , stream?.CvPtr ?? Stream.Null.CvPtr);
                h_lines = vec.ToArray<LineSegmentPoint>();
            }

            GC.KeepAlive(this);
            GC.KeepAlive(d_lines);
            GC.KeepAlive(h_lines);
        }

        #region Properties

        /// <summary>
        /// 
        /// </summary>
        public float Rho {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_HoughSegmentDetector_getRho(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_HoughSegmentDetector_setRho(ptr, value);
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public float Theta {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_HoughSegmentDetector_getTheta(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_HoughSegmentDetector_setTheta(ptr, value);
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public int minLineLength {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_HoughSegmentDetector_getMinLineLength(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_HoughSegmentDetector_setMinLineLength(ptr, value);
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public int maxLineGap {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_HoughSegmentDetector_getMaxLineGap(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_HoughSegmentDetector_setMaxLineGap(ptr, value);
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public int MaxLines {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_HoughSegmentDetector_getMaxLines(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_HoughSegmentDetector_setMaxLines(ptr, value);
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