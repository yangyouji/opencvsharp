using System;

namespace OpenCvSharp.Cuda {
    // ReSharper disable InconsistentNaming

    /// <summary>
    /// Base class for lines detector algorithm. :
    /// </summary>
    public class HoughLinesDetector : Algorithm {
        /// <summary>
        /// cv::Ptr&lt;T&gt;
        /// </summary>
        private Ptr objectPtr;

        #region Init & Disposal

        /// <summary>
        /// Creates implementation for cuda::HoughLinesDetector .
        /// </summary>
        /// <param name="rho">Distance resolution of the accumulator in pixels.</param>
        /// <param name="theta">Angle resolution of the accumulator in radians.</param>
        /// <param name="threshold">Accumulator threshold parameter. Only those lines are returned that get enough
        /// votes( \f$>\texttt{ threshold}\f$ ).</param>
        /// <param name="doSort">Performs lines sort by votes.</param>
        /// <param name="maxLines">Maximum number of output lines.</param>
        /// <returns></returns>
        public static HoughLinesDetector create(float rho, float theta, int threshold, bool doSort = false, int maxLines = 4096) {
            IntPtr ptr = NativeMethods.cuda_imgproc_createHoughLinesDetector(
                rho, theta, threshold, doSort, maxLines);
            return new HoughLinesDetector(ptr);
        }

        internal HoughLinesDetector(IntPtr ptr) {
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
        /// Finds lines in a binary image using the classical Hough transform.
        /// </summary>
        /// <param name="src">8-bit, single-channel binary source image.</param>
        /// <param name="lines">Output vector of lines. Each line is represented by a two-element vector
        /// \f$(\rho, \theta)\f$ . \f$\rho\f$ is the distance from the coordinate origin \f$(0,0)\f$ (top-left corner of
        /// the image). \f$\theta\f$ is the line rotation angle in radians(
        /// \f$0 \sim \textrm{ vertical line}, \pi/2 \sim \textrm{horizontal line}\f$ ).</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public virtual void detect(InputArray src, OutputArray lines, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (lines == null)
                throw new ArgumentNullException(nameof(lines));
            src.ThrowIfDisposed();
            lines.ThrowIfNotReady();

            NativeMethods.cuda_imgproc_HoughLinesDetector_detect(ptr, src.CvPtr, lines.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);

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
        /// <param name="h_votes">Optional output array for line's votes.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public virtual void downloadResults(InputArray d_lines, out LineSegmentPolar[] h_lines, OutputArray h_votes = null, Stream stream = null) {
            if (d_lines == null)
                throw new ArgumentNullException(nameof(d_lines));
            d_lines.ThrowIfDisposed();

            using (var vec = new VectorOfVec2f()) {
                NativeMethods.cuda_imgproc_HoughLinesDetector_downloadResults(ptr, d_lines.CvPtr, vec.CvPtr
                    , h_votes?.CvPtr ?? IntPtr.Zero, stream?.CvPtr ?? Stream.Null.CvPtr);
                h_lines = vec.ToArray<LineSegmentPolar>();
            }

            GC.KeepAlive(this);
            GC.KeepAlive(d_lines);
            GC.KeepAlive(h_lines);
            GC.KeepAlive(h_votes);
            h_votes?.Fix();
        }


        #region Properties

        /// <summary>
        /// 
        /// </summary>
        public float Rho {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_HoughLinesDetector_getRho(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_HoughLinesDetector_setRho(ptr, value);
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public float Theta {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_HoughLinesDetector_getTheta(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_HoughLinesDetector_setTheta(ptr, value);
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public int Threshold {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_HoughLinesDetector_getThreshold(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_HoughLinesDetector_setThreshold(ptr, value);
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public bool DoSort {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_HoughLinesDetector_getDoSort(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_HoughLinesDetector_setDoSort(ptr, value);
                GC.KeepAlive(this);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public int MaxLines {
            get {
                ThrowIfDisposed();
                var res = NativeMethods.cuda_imgproc_HoughLinesDetector_getMaxLines(ptr);
                GC.KeepAlive(this);
                return res;
            }
            set {
                ThrowIfDisposed();
                NativeMethods.cuda_imgproc_HoughLinesDetector_setMaxLines(ptr, value);
                GC.KeepAlive(this);
            }
        }

        #endregion

        internal class Ptr : OpenCvSharp.Ptr {
            public Ptr(IntPtr ptr) : base(ptr) {
            }

            public override IntPtr Get() {
                var res = NativeMethods.cuda_imgproc_Ptr_HoughLinesDetector_get(ptr);
                GC.KeepAlive(this);
                return res;
            }

            protected override void DisposeUnmanaged() {
                NativeMethods.cuda_imgproc_Ptr_HoughLinesDetector_delete(ptr);
                base.DisposeUnmanaged();
            }
        }
    }
}