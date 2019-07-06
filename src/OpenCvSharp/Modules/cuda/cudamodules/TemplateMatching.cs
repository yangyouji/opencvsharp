using System;

namespace OpenCvSharp.Cuda {
    // ReSharper disable InconsistentNaming

    /// <summary>
    /// Base class for Template Matching. :
    /// </summary>
    public class TemplateMatching : Algorithm {
        /// <summary>
        /// cv::Ptr&lt;T&gt;
        /// </summary>
        private Ptr objectPtr;

        #region Init & Disposal

        /// <summary>
        /// Creates implementation for cuda::TemplateMatching .
        /// </summary>
        /// <param name="srcType">Input source type. CV_32F and CV_8U depth images (1..4 channels) are supported
        /// for now.</param>
        /// <param name="method">Specifies the way to compare the template with the image.
        /// The following methods are supported for the CV_8U depth images for now:
        /// -   CV_TM_SQDIFF
        /// -   CV_TM_SQDIFF_NORMED
        /// -   CV_TM_CCORR
        /// -   CV_TM_CCORR_NORMED
        /// -   CV_TM_CCOEFF
        /// -   CV_TM_CCOEFF_NORMED
        /// The following methods are supported for the CV_32F images for now:
        /// -   CV_TM_SQDIFF
        /// -   CV_TM_CCORR</param>
        /// <param name="user_block_size">You can use field user_block_size to set specific block size. If you
        /// leave its default value Size(0,0) then automatic estimation of block size will be used(which is
        /// optimized for speed). By varying user_block_size you can reduce memory requirements at the cost
        /// of speed.</param>
        /// <returns></returns>
        public static TemplateMatching create(int srcType, TemplateMatchModes method, Size? user_block_size = null) {
            IntPtr ptr = NativeMethods.cuda_imgproc_createTemplateMatching(
                srcType, (int)method, user_block_size.GetValueOrDefault(new Size(0,0)));
            return new TemplateMatching(ptr);
        }

        internal TemplateMatching(IntPtr ptr) {
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
        /// Computes a proximity map for a raster template and an image where the template is searched for.
        /// </summary>
        /// <param name="image">Source image.</param>
        /// <param name="templ">Template image with the size and type the same as image .</param>
        /// <param name="result">Map containing comparison results ( CV_32FC1 ). If image is *W x H* and templ is *w
        /// x h*, then result must be* W-w+1 x H-h+1*.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public virtual void match(InputArray image, InputArray templ, OutputArray result, Stream stream = null) {
            if (image == null)
                throw new ArgumentNullException(nameof(image));
            if (templ == null)
                throw new ArgumentNullException(nameof(templ));
            if (result == null)
                throw new ArgumentNullException(nameof(result));
            image.ThrowIfDisposed();
            templ.ThrowIfDisposed();
            result.ThrowIfNotReady();

            NativeMethods.cuda_imgproc_CornersDetector_detect(ptr, image.CvPtr, templ.CvPtr, result.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);

            result.Fix();
            GC.KeepAlive(this);
            GC.KeepAlive(image);
            GC.KeepAlive(templ);
            GC.KeepAlive(result);
        }

        ///// <summary>
        ///// Downloads results from cuda::CornersDetector::detect to host memory.
        ///// </summary>
        ///// <param name="d_corners">Result of cuda::CornersDetector::detect .</param>
        ///// <param name="c_corners">Output host array.</param>
        ///// <param name="stream">Stream for the asynchronous version.</param>
        //public virtual void downloadResults(InputArray d_corners, out Point2f[] c_corners, Stream stream = null) {
        //    if (d_corners == null)
        //        throw new ArgumentNullException(nameof(d_corners));
        //    d_corners.ThrowIfDisposed();

        //    using (var vec = new VectorOfVec2f()) {
        //        NativeMethods.cuda_imgproc_CornersDetector_downloadResults(ptr, d_corners.CvPtr, vec.CvPtr
        //            , stream?.CvPtr ?? Stream.Null.CvPtr);
        //        c_corners = vec.ToArray<Point2f>();
        //    }

        //    GC.KeepAlive(this);
        //    GC.KeepAlive(d_corners);
        //    GC.KeepAlive(c_corners);
        //}

        internal class Ptr : OpenCvSharp.Ptr {
            public Ptr(IntPtr ptr) : base(ptr) {
            }

            public override IntPtr Get() {
                var res = NativeMethods.cuda_imgproc_Ptr_TemplateMatching_get(ptr);
                GC.KeepAlive(this);
                return res;
            }

            protected override void DisposeUnmanaged() {
                NativeMethods.cuda_imgproc_Ptr_TemplateMatching_delete(ptr);
                base.DisposeUnmanaged();
            }
        }
    }
}