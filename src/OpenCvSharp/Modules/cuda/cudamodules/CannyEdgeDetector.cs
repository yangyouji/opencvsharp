using System;

namespace OpenCvSharp.Cuda {
    // ReSharper disable InconsistentNaming

    /// <summary>
    /// Creates implementation for cuda::CannyEdgeDetector
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
        public static CannyEdgeDetector Create(
            double low_thresh, double high_thresh, int apperture_size = 3, bool L2gradient = false) {
            IntPtr ptr = NativeMethods.cuda_createCannyEdgeDetector(
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
        /// <param name="image"></param>
        /// <param name="edges"></param>
        /// <param name="stream"></param>
        public virtual void detect(InputArray image, OutputArray edges, Stream stream = null) {
            if (image == null)
                throw new ArgumentNullException(nameof(image));
            if (edges == null)
                throw new ArgumentNullException(nameof(edges));
            image.ThrowIfDisposed();
            edges.ThrowIfNotReady();

            NativeMethods.cuda_CannyEdgeDetector_detect(ptr, image.CvPtr, edges.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);

            edges.Fix();
            GC.KeepAlive(this);
            GC.KeepAlive(image);
            GC.KeepAlive(edges);
        }


        //#region Properties

        ///// <summary>
        ///// 
        ///// </summary>
        //public int History {
        //    get {
        //        ThrowIfDisposed();
        //        var res = NativeMethods.bgsegm_BackgroundSubtractorMOG_getHistory(ptr);
        //        GC.KeepAlive(this);
        //        return res;
        //    }
        //    set {
        //        ThrowIfDisposed();
        //        NativeMethods.bgsegm_BackgroundSubtractorMOG_setHistory(ptr, value);
        //        GC.KeepAlive(this);
        //    }
        //}

        ///// <summary>
        ///// 
        ///// </summary>
        //public int NMixtures {
        //    get {
        //        ThrowIfDisposed();
        //        var res = NativeMethods.bgsegm_BackgroundSubtractorMOG_getNMixtures(ptr);
        //        GC.KeepAlive(this);
        //        return res;
        //    }
        //    set {
        //        ThrowIfDisposed();
        //        NativeMethods.bgsegm_BackgroundSubtractorMOG_setNMixtures(ptr, value);
        //        GC.KeepAlive(this);
        //    }
        //}

        ///// <summary>
        ///// 
        ///// </summary>
        //public double BackgroundRatio {
        //    get {
        //        ThrowIfDisposed();
        //        var res = NativeMethods.bgsegm_BackgroundSubtractorMOG_getBackgroundRatio(ptr);
        //        GC.KeepAlive(this);
        //        return res;
        //    }
        //    set {
        //        ThrowIfDisposed();
        //        NativeMethods.bgsegm_BackgroundSubtractorMOG_setBackgroundRatio(ptr, value);
        //        GC.KeepAlive(this);
        //    }
        //}

        ///// <summary>
        ///// 
        ///// </summary>
        //public double NoiseSigma {
        //    get {
        //        ThrowIfDisposed();
        //        var res = NativeMethods.bgsegm_BackgroundSubtractorMOG_getNoiseSigma(ptr);
        //        GC.KeepAlive(this);
        //        return res;
        //    }
        //    set {
        //        ThrowIfDisposed();
        //        NativeMethods.bgsegm_BackgroundSubtractorMOG_setNoiseSigma(ptr, value);
        //        GC.KeepAlive(this);
        //    }
        //}

        //#endregion

        internal class Ptr : OpenCvSharp.Ptr {
            public Ptr(IntPtr ptr) : base(ptr) {
            }

            public override IntPtr Get() {
                var res = NativeMethods.cuda_Ptr_CannyEdgeDetector_get(ptr);
                GC.KeepAlive(this);
                return res;
            }

            protected override void DisposeUnmanaged() {
                NativeMethods.cuda_Ptr_CannyEdgeDetector_delete(ptr);
                base.DisposeUnmanaged();
            }
        }
    }
}