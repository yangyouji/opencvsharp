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
            var ret = NativeMethods.cuda_arithm_countNonZero(mtx.CvPtr);
            GC.KeepAlive(mtx);
            return ret;
        }
        #endregion

        #region cuda Add
        /// <summary>
        /// Computes a matrix-matrix or matrix-scalar sum.
        /// </summary>
        /// <param name="src1">First source matrix or scalar.</param>
        /// <param name="src2">Second source matrix or scalar. Matrix should have the same size and type as src1 .</param>
        /// <param name="dst">Destination matrix that has the same size and number of channels as the input array(s).
        /// The depth is defined by dtype or src1 depth.</param>
        /// <param name="mask">Optional operation mask, 8-bit single channel array, that specifies elements of the
        /// destination array to be changed.The mask can be used only with single channel images.</param>
        /// <param name="dtype">Optional depth of the output array.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void add(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = null, int dtype = -1, Stream stream = null) {
            if (src1 == null)
                throw new ArgumentNullException(nameof(src1));
            if (src2 == null)
                throw new ArgumentNullException(nameof(src2));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_add(src1.CvPtr, src2.CvPtr, dst.CvPtr, mask?.CvPtr ?? IntPtr.Zero, dtype, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            GC.KeepAlive(dst);
            GC.KeepAlive(mask);
            dst.Fix();
        }

        #endregion

        #region cuda subtract
        /// <summary>
        /// Computes a matrix-matrix or matrix-scalar difference.
        /// </summary>
        /// <param name="src1">First source matrix or scalar.</param>
        /// <param name="src2">Second source matrix or scalar. Matrix should have the same size and type as src1 .</param>
        /// <param name="dst">Destination matrix that has the same size and number of channels as the input array(s).
        /// The depth is defined by dtype or src1 depth.</param>
        /// <param name="mask">Optional operation mask, 8-bit single channel array, that specifies elements of the
        /// destination array to be changed.The mask can be used only with single channel images.</param>
        /// <param name="dtype">Optional depth of the output array.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void subtract(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = null, int dtype = -1, Stream stream = null) {
            if (src1 == null)
                throw new ArgumentNullException(nameof(src1));
            if (src2 == null)
                throw new ArgumentNullException(nameof(src2));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_subtract(src1.CvPtr, src2.CvPtr, dst.CvPtr, mask?.CvPtr ?? IntPtr.Zero, dtype, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            GC.KeepAlive(dst);
            GC.KeepAlive(mask);
            dst.Fix();
        }

        #endregion

        #region cuda multiply
        /// <summary>
        /// Computes a matrix-matrix or matrix-scalar per-element product.
        /// </summary>
        /// <param name="src1">First source matrix or scalar.</param>
        /// <param name="src2">Second source matrix or scalar.</param>
        /// <param name="dst">Destination matrix that has the same size and number of channels as the input array(s).
        /// The depth is defined by dtype or src1 depth.</param>
        /// <param name="scale">Optional scale factor.</param>
        /// <param name="dtype">Optional depth of the output array.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void multiply(InputArray src1, InputArray src2, OutputArray dst, double scale = 1, int dtype = -1, Stream stream = null) {
            if (src1 == null)
                throw new ArgumentNullException(nameof(src1));
            if (src2 == null)
                throw new ArgumentNullException(nameof(src2));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_multiply(src1.CvPtr, src2.CvPtr, dst.CvPtr, scale, dtype, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda divide
        /// <summary>
        /// Computes a matrix-matrix or matrix-scalar division.
        /// </summary>
        /// <param name="src1">First source matrix or scalar.</param>
        /// <param name="src2">Second source matrix or scalar.</param>
        /// <param name="dst">Destination matrix that has the same size and number of channels as the input array(s).
        /// The depth is defined by dtype or src1 depth.</param>
        /// <param name="scale">Optional scale factor.</param>
        /// <param name="dtype">Optional depth of the output array.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void divide(InputArray src1, InputArray src2, OutputArray dst, double scale = 1, int dtype = -1, Stream stream = null) {
            if (src1 == null)
                throw new ArgumentNullException(nameof(src1));
            if (src2 == null)
                throw new ArgumentNullException(nameof(src2));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_divide(src1.CvPtr, src2.CvPtr, dst.CvPtr, scale, dtype, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda absdiff
        /// <summary>
        /// Computes per-element absolute difference of two matrices (or of a matrix and scalar).
        /// </summary>
        /// <param name="src1">First source matrix or scalar.</param>
        /// <param name="src2">Second source matrix or scalar.</param>
        /// <param name="dst">Destination matrix that has the same size and number of channels as the input array(s).
        /// The depth is defined by dtype or src1 depth.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void absdiff(InputArray src1, InputArray src2, OutputArray dst, Stream stream = null) {
            if (src1 == null)
                throw new ArgumentNullException(nameof(src1));
            if (src2 == null)
                throw new ArgumentNullException(nameof(src2));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_absdiff(src1.CvPtr, src2.CvPtr, dst.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda abs
        /// <summary>
        /// Computes an absolute value of each matrix element.
        /// </summary>
        /// <param name="src">Source matrix.</param>
        /// <param name="dst">Destination matrix with the same size and type as src .</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void abs(InputArray src, OutputArray dst, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_abs(src.CvPtr, dst.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda sqr
        /// <summary>
        /// Computes a square value of each matrix element.
        /// </summary>
        /// <param name="src">Source matrix.</param>
        /// <param name="dst">Destination matrix with the same size and type as src .</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void sqr(InputArray src, OutputArray dst, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_sqr(src.CvPtr, dst.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda sqrt
        /// <summary>
        /// Computes a square root of each matrix element.
        /// </summary>
        /// <param name="src">Source matrix.</param>
        /// <param name="dst">Destination matrix with the same size and type as src .</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void sqrt(InputArray src, OutputArray dst, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_sqrt(src.CvPtr, dst.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda exp
        /// <summary>
        /// Computes an exponent of each matrix element.
        /// </summary>
        /// <param name="src">Source matrix.</param>
        /// <param name="dst">Destination matrix with the same size and type as src .</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void exp(InputArray src, OutputArray dst, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_exp(src.CvPtr, dst.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda log
        /// <summary>
        /// Computes a natural logarithm of absolute value of each matrix element.
        /// </summary>
        /// <param name="src">Source matrix.</param>
        /// <param name="dst">Destination matrix with the same size and type as src .</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void log(InputArray src, OutputArray dst, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_log(src.CvPtr, dst.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda pow
        /// <summary>
        /// Raises every matrix element to a power.
        /// </summary>
        /// <param name="src">Source matrix.</param>
        /// <param name="power">Exponent of power.</param>
        /// <param name="dst">Destination matrix with the same size and type as src .</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void pow(InputArray src, double power, OutputArray dst, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_pow(src.CvPtr, power, dst.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda compare
        /// <summary>
        /// Compares elements of two matrices (or of a matrix and scalar).
        /// </summary>
        /// <param name="src1">First source matrix or scalar.</param>
        /// <param name="src2">Second source matrix or scalar.</param>
        /// <param name="dst">Destination matrix that has the same size as the input array(s) and type CV_8U.</param>
        /// <param name="cmpop">Flag specifying the relation between the elements to be checked:
        /// -   **CMP_EQ:** a(.) == b(.)
        /// -   **CMP_GT:** a(.) \> b(.)
        /// -   **CMP_GE:** a(.) \>= b(.)
        /// -   **CMP_LT:** a(.) \< b(.)\
        /// -   **CMP_LE:** a(.) \<= b(.)
        /// -   **CMP_NE:** a(.) != b(.)</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void compare(InputArray src1, InputArray src2, OutputArray dst, CmpTypes cmpop, Stream stream = null) {
            if (src1 == null)
                throw new ArgumentNullException(nameof(src1));
            if (src2 == null)
                throw new ArgumentNullException(nameof(src2));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_compare(src1.CvPtr, src2.CvPtr, dst.CvPtr, (int)cmpop, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda bitwise_not
        /// <summary>
        /// Performs a per-element bitwise inversion.
        /// </summary>
        /// <param name="src">Source matrix.</param>
        /// <param name="dst">Destination matrix with the same size and type as src .</param>
        /// <param name="mask">Optional operation mask, 8-bit single channel array, that specifies elements of the
        /// destination array to be changed.The mask can be used only with single channel images.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void bitwise_not(InputArray src,OutputArray dst, InputArray mask = null, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_bitwise_not(src.CvPtr, dst.CvPtr, mask?.CvPtr ?? IntPtr.Zero, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            GC.KeepAlive(mask);
            dst.Fix();
        }
        #endregion

        #region cuda bitwise_or
        /// <summary>
        /// Performs a per-element bitwise disjunction of two matrices (or of matrix and scalar).
        /// </summary>
        /// <param name="src1">First source matrix or scalar.</param>
        /// <param name="src2">Second source matrix or scalar.</param>
        /// <param name="dst">Destination matrix that has the same size and type as the input array(s).</param>
        /// <param name="mask">Optional operation mask, 8-bit single channel array, that specifies elements of the
        /// destination array to be changed.The mask can be used only with single channel images.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void bitwise_or(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = null, Stream stream = null) {
            if (src1 == null)
                throw new ArgumentNullException(nameof(src1));
            if (src2 == null)
                throw new ArgumentNullException(nameof(src2));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_bitwise_or(src1.CvPtr, src2.CvPtr, dst.CvPtr, mask?.CvPtr ?? IntPtr.Zero, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            GC.KeepAlive(dst);
            GC.KeepAlive(mask);
            dst.Fix();
        }
        #endregion

        #region cuda bitwise_and
        /// <summary>
        /// Performs a per-element bitwise conjunction of two matrices (or of matrix and scalar).
        /// </summary>
        /// <param name="src1">First source matrix or scalar.</param>
        /// <param name="src2">Second source matrix or scalar.</param>
        /// <param name="dst">Destination matrix that has the same size and type as the input array(s).</param>
        /// <param name="mask">Optional operation mask, 8-bit single channel array, that specifies elements of the
        /// destination array to be changed.The mask can be used only with single channel images.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void bitwise_and(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = null, Stream stream = null) {
            if (src1 == null)
                throw new ArgumentNullException(nameof(src1));
            if (src2 == null)
                throw new ArgumentNullException(nameof(src2));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_bitwise_and(src1.CvPtr, src2.CvPtr, dst.CvPtr, mask?.CvPtr ?? IntPtr.Zero, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            GC.KeepAlive(dst);
            GC.KeepAlive(mask);
            dst.Fix();
        }
        #endregion

        #region cuda bitwise_xor
        /// <summary>
        /// Performs a per-element bitwise exclusive or operation of two matrices (or of matrix and scalar).
        /// </summary>
        /// <param name="src1">First source matrix or scalar.</param>
        /// <param name="src2">Second source matrix or scalar.</param>
        /// <param name="dst">Destination matrix that has the same size and type as the input array(s).</param>
        /// <param name="mask">Optional operation mask, 8-bit single channel array, that specifies elements of the
        /// destination array to be changed.The mask can be used only with single channel images.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void bitwise_xor(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = null, Stream stream = null) {
            if (src1 == null)
                throw new ArgumentNullException(nameof(src1));
            if (src2 == null)
                throw new ArgumentNullException(nameof(src2));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_bitwise_xor(src1.CvPtr, src2.CvPtr, dst.CvPtr, mask?.CvPtr ?? IntPtr.Zero, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            GC.KeepAlive(dst);
            GC.KeepAlive(mask);
            dst.Fix();
        }
        #endregion

        #region cuda min
        /// <summary>
        /// Computes the per-element minimum of two matrices (or a matrix and a scalar).
        /// </summary>
        /// <param name="src1">First source matrix or scalar.</param>
        /// <param name="src2">Second source matrix or scalar.</param>
        /// <param name="dst">Destination matrix that has the same size and type as the input array(s).</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void min(InputArray src1, InputArray src2, OutputArray dst, Stream stream = null) {
            if (src1 == null)
                throw new ArgumentNullException(nameof(src1));
            if (src2 == null)
                throw new ArgumentNullException(nameof(src2));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_min(src1.CvPtr, src2.CvPtr, dst.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda max
        /// <summary>
        /// Computes the per-element maximum of two matrices (or a matrix and a scalar).
        /// </summary>
        /// <param name="src1">First source matrix or scalar.</param>
        /// <param name="src2">Second source matrix or scalar.</param>
        /// <param name="dst">Destination matrix that has the same size and type as the input array(s).</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void max(InputArray src1, InputArray src2, OutputArray dst, Stream stream = null) {
            if (src1 == null)
                throw new ArgumentNullException(nameof(src1));
            if (src2 == null)
                throw new ArgumentNullException(nameof(src2));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_max(src1.CvPtr, src2.CvPtr, dst.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda addWeighted
        /// <summary>
        /// Computes the weighted sum of two arrays.
        /// </summary>
        /// <param name="src1">First source array.</param>
        /// <param name="alpha">Weight for the first array elements.</param>
        /// <param name="src2">Second source array of the same size and channel number as src1 .</param>
        /// <param name="beta">Weight for the second array elements.</param>
        /// <param name="gamma">Scalar added to each sum.</param>
        /// <param name="dst">Destination array that has the same size and number of channels as the input arrays.</param>
        /// <param name="dtype">Optional depth of the destination array. When both input arrays have the same depth,
        /// dtype can be set to -1, which will be equivalent to src1.depth().</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void addWeighted(InputArray src1, double alpha, InputArray src2, double beta, double gamma, OutputArray dst, int dtype = -1, Stream stream = null) {
            if (src1 == null)
                throw new ArgumentNullException(nameof(src1));
            if (src2 == null)
                throw new ArgumentNullException(nameof(src2));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_addWeighted(src1.CvPtr, alpha, src2.CvPtr, beta, gamma, dst.CvPtr, dtype, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda scaleAdd
        /// <summary>
        /// adds scaled array to another one (dst = alpha*src1 + src2)
        /// </summary>
        /// <param name="src1">First source array.</param>
        /// <param name="alpha">Weight for the first array elements.</param>
        /// <param name="src2">Second source array of the same size and channel number as src1 .</param>
        /// <param name="dst">Destination array that has the same size and number of channels as the input arrays.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void scaleAdd(InputArray src1, double alpha, InputArray src2, OutputArray dst, Stream stream = null) {
            addWeighted(src1, alpha, src2, 1.0, 0.0, dst, -1, stream);
        }
        #endregion

        #region cuda threshold
        /// <summary>
        /// Applies a fixed-level threshold to each array element.
        /// </summary>
        /// <param name="src">Source array (single-channel).</param>
        /// <param name="dst">Destination array with the same size and type as src .</param>
        /// <param name="thresh">Threshold value.</param>
        /// <param name="maxval">Maximum value to use with THRESH_BINARY and THRESH_BINARY_INV threshold types.</param>
        /// <param name="type">Threshold type. For details, see threshold . The THRESH_OTSU and THRESH_TRIANGLE
        /// threshold types are not supported.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void threshold(InputArray src, OutputArray dst, double thresh, double maxval, ThresholdTypes type, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_arithm_threshold(src.CvPtr, dst.CvPtr, thresh, maxval, (int)type, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda magnitude
        /// <summary>
        /// Computes magnitudes of complex matrix elements.
        /// </summary>
        /// <param name="xy">Source complex matrix in the interleaved format ( CV_32FC2 ).</param>
        /// <param name="magnitude">Destination matrix of float magnitudes ( CV_32FC1 ).</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void magnitude(InputArray xy, OutputArray magnitude, Stream stream = null) {
            if (xy == null)
                throw new ArgumentNullException(nameof(xy));
            if (magnitude == null)
                throw new ArgumentNullException(nameof(magnitude));
            xy.ThrowIfDisposed();
            magnitude.ThrowIfNotReady();
            NativeMethods.cuda_arithm_magnitude_0(xy.CvPtr, magnitude.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(xy);
            GC.KeepAlive(magnitude);
            magnitude.Fix();
        }
        #endregion

        #region cuda magnitudeSqr
        /// <summary>
        /// Computes squared magnitudes of complex matrix elements.
        /// </summary>
        /// <param name="xy">Source complex matrix in the interleaved format ( CV_32FC2 ).</param>
        /// <param name="magnitude">Destination matrix of float magnitude squares ( CV_32FC1 ).</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void magnitudeSqr(InputArray xy, OutputArray magnitude, Stream stream = null) {
            if (xy == null)
                throw new ArgumentNullException(nameof(xy));
            if (magnitude == null)
                throw new ArgumentNullException(nameof(magnitude));
            xy.ThrowIfDisposed();
            magnitude.ThrowIfNotReady();
            NativeMethods.cuda_arithm_magnitudeSqr_0(xy.CvPtr, magnitude.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(xy);
            GC.KeepAlive(magnitude);
            magnitude.Fix();
        }
        #endregion

        #region cuda magnitude
        /// <summary>
        /// computes magnitude of each (x(i), y(i)) vector
        /// supports only floating-point source
        /// </summary>
        /// <param name="x">Source matrix containing real components ( CV_32FC1 ).</param>
        /// <param name="y">Source matrix containing imaginary components ( CV_32FC1 ).</param>
        /// <param name="magnitude">Destination matrix of float magnitudes ( CV_32FC1 ).</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void magnitude(InputArray x, InputArray y, OutputArray magnitude, Stream stream = null) {
            if (x == null)
                throw new ArgumentNullException(nameof(x));
            if (y == null)
                throw new ArgumentNullException(nameof(y));
            if (magnitude == null)
                throw new ArgumentNullException(nameof(magnitude));
            x.ThrowIfDisposed();
            y.ThrowIfDisposed();
            magnitude.ThrowIfNotReady();
            NativeMethods.cuda_arithm_magnitude_1(x.CvPtr, y.CvPtr, magnitude.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(x);
            GC.KeepAlive(y);
            GC.KeepAlive(magnitude);
            magnitude.Fix();
        }
        #endregion

        #region cuda magnitudeSqr
        /// <summary>
        /// computes squared magnitude of each (x(i), y(i)) vector
        /// supports only floating-point source
        /// </summary>
        /// <param name="x">Source matrix containing real components ( CV_32FC1 ).</param>
        /// <param name="y">Source matrix containing imaginary components ( CV_32FC1 ).</param>
        /// <param name="magnitude">Destination matrix of float magnitude squares ( CV_32FC1 ).</param>
        /// <param name="stream">Stream for the asynchronous version.</param>

        public static void magnitudeSqr(InputArray x, InputArray y, OutputArray magnitude, Stream stream = null) {
            if (x == null)
                throw new ArgumentNullException(nameof(x));
            if (y == null)
                throw new ArgumentNullException(nameof(y));
            if (magnitude == null)
                throw new ArgumentNullException(nameof(magnitude));
            x.ThrowIfDisposed();
            y.ThrowIfDisposed();
            magnitude.ThrowIfNotReady();
            NativeMethods.cuda_arithm_magnitudeSqr_1(x.CvPtr, y.CvPtr, magnitude.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(x);
            GC.KeepAlive(y);
            GC.KeepAlive(magnitude);
            magnitude.Fix();
        }
        #endregion






    }
}
