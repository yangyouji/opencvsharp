using System;

namespace OpenCvSharp.Cuda {
    // ReSharper disable InconsistentNaming

    /// <summary>
    /// Determines strong corners on an image.
    /// </summary>
    public class CornersDetector : Algorithm {
        /// <summary>
        /// cv::Ptr&lt;T&gt;
        /// </summary>
        private Ptr objectPtr;

        #region Init & Disposal

        /// <summary>
        /// Creates implementation for cuda::CornersDetector .
        /// </summary>
        /// <param name="srcType">Input source type. Only CV_8UC1 and CV_32FC1 are supported for now.</param>
        /// <param name="maxCorners">Maximum number of corners to return. If there are more corners than are found,
        /// the strongest of them is returned.</param>
        /// <param name="qualityLevel">Parameter characterizing the minimal accepted quality of image corners. The
        /// parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue
        /// (see cornerMinEigenVal) or the Harris function response(see cornerHarris). The corners with the
        /// quality measure less than the product are rejected.For example, if the best corner has the
        /// quality measure = 1500, and the qualityLevel= 0.01, then all the corners with the quality measure
        /// less than 15 are rejected.</param>
        /// <param name="minDistance">Minimum possible Euclidean distance between the returned corners.</param>
        /// <param name="blockSize">Size of an average block for computing a derivative covariation matrix over each
        /// pixel neighborhood.See cornerEigenValsAndVecs.</param>
        /// <param name="useHarrisDetector">Parameter indicating whether to use a Harris detector (see cornerHarris)
        /// or cornerMinEigenVal.</param>
        /// <param name="harrisK">Free parameter of the Harris detector.</param>
        /// <returns></returns>
        public static CornersDetector create(int srcType, int maxCorners = 1000, double qualityLevel = 0.01, double minDistance = 0.0,
                                                                  int blockSize = 3, bool useHarrisDetector = false, double harrisK = 0.04) {
            IntPtr ptr = NativeMethods.cuda_imgproc_createGoodFeaturesToTrackDetector(
                srcType, maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, harrisK);
            return new CornersDetector(ptr);
        }

        internal CornersDetector(IntPtr ptr) {
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
        /// Determines strong corners on an image.
        /// </summary>
        /// <param name="image">Input 8-bit or floating-point 32-bit, single-channel image.</param>
        /// <param name="corners">Output vector of detected corners (1-row matrix with CV_32FC2 type with corners
        /// positions).</param>
        /// <param name="mask">Optional region of interest. If the image is not empty (it needs to have the type
        /// CV_8UC1 and the same size as image ), it specifies the region in which the corners are detected.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public virtual void detect(InputArray image, OutputArray corners, InputArray mask = null, Stream stream = null) {
            if (image == null)
                throw new ArgumentNullException(nameof(image));
            if (corners == null)
                throw new ArgumentNullException(nameof(corners));
            image.ThrowIfDisposed();
            corners.ThrowIfNotReady();

            NativeMethods.cuda_imgproc_CornersDetector_detect(ptr, image.CvPtr, corners.CvPtr, mask?.CvPtr ?? IntPtr.Zero, stream?.CvPtr ?? Stream.Null.CvPtr);

            corners.Fix();
            GC.KeepAlive(this);
            GC.KeepAlive(image);
            GC.KeepAlive(corners);
        }

        /// <summary>
        /// Downloads results from cuda::CornersDetector::detect to host memory.
        /// </summary>
        /// <param name="d_corners">Result of cuda::CornersDetector::detect .</param>
        /// <param name="c_corners">Output host array.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public virtual void downloadResults(InputArray d_corners, out Point2f[] c_corners, Stream stream = null) {
            if (d_corners == null)
                throw new ArgumentNullException(nameof(d_corners));
            d_corners.ThrowIfDisposed();

            using (var vec = new VectorOfVec2f()) {
                NativeMethods.cuda_imgproc_CornersDetector_downloadResults(ptr, d_corners.CvPtr, vec.CvPtr
                    , stream?.CvPtr ?? Stream.Null.CvPtr);
                c_corners = vec.ToArray<Point2f>();
            }

            GC.KeepAlive(this);
            GC.KeepAlive(d_corners);
            GC.KeepAlive(c_corners);
        }

        internal class Ptr : OpenCvSharp.Ptr {
            public Ptr(IntPtr ptr) : base(ptr) {
            }

            public override IntPtr Get() {
                var res = NativeMethods.cuda_imgproc_Ptr_CornersDetector_get(ptr);
                GC.KeepAlive(this);
                return res;
            }

            protected override void DisposeUnmanaged() {
                NativeMethods.cuda_imgproc_Ptr_CornersDetector_delete(ptr);
                base.DisposeUnmanaged();
            }
        }
    }
}