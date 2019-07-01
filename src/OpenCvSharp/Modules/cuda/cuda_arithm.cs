using System;
using System.Collections.Generic;
using System.Text;

namespace OpenCvSharp.Cuda {

    /// <summary>
    ///  GPU warping
    /// </summary>
    public static partial class cuda {

        #region cuda CountNonZero
        /// <summary>
        /// Counts non-zero matrix elements
        /// </summary>
        /// <param name="mtx">Single-channel source image</param>
        /// <returns>number of non-zero elements in mtx</returns>
        public static int CountNonZero(InputArray mtx) {
            if (mtx == null)
                throw new ArgumentNullException(nameof(mtx));
            mtx.ThrowIfDisposed();
            var ret = NativeMethods.cuda_countNonZero(mtx.CvPtr);
            GC.KeepAlive(mtx);
            return ret;
        }
        #endregion

    }
}
