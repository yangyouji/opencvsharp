using System;
using System.Collections.Generic;
using System.Text;

namespace OpenCvSharp.Cuda {

    /// <summary>
    ///  GPU warping
    /// </summary>
    public static partial class cuda {

        #region cuda Remap
        /// <summary>
        /// Applies a generic geometrical transformation to an image.
        /// </summary>
        /// <param name="src">Source image.</param>
        /// <param name="dst">Destination image with the size the same as xmap and the type the same as src .</param>
        /// <param name="map1">X values. Only CV_32FC1 type is supported.</param>
        /// <param name="map2">Y values. Only CV_32FC1 type is supported.</param>
        /// <param name="interpolation">Interpolation method (see resize ). INTER_NEAREST , INTER_LINEAR and
        /// INTER_CUBIC are supported for now.</param>
        /// <param name="borderMode">Pixel extrapolation method (see borderInterpolate ). BORDER_REFLECT101 ,
        /// BORDER_REPLICATE , BORDER_CONSTANT , BORDER_REFLECT and BORDER_WRAP are supported for now.</param>
        /// <param name="borderValue">Value used in case of a constant border. By default, it is 0.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void remap(
            InputArray src, OutputArray dst, InputArray map1, InputArray map2,
            InterpolationFlags interpolation = InterpolationFlags.Linear,
            BorderTypes borderMode = BorderTypes.Constant, Scalar? borderValue = null, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            if (map1 == null)
                throw new ArgumentNullException(nameof(map1));
            if (map2 == null)
                throw new ArgumentNullException(nameof(map2));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            map1.ThrowIfDisposed();
            map2.ThrowIfDisposed();
            Scalar borderValue0 = borderValue.GetValueOrDefault(Scalar.All(0));
            NativeMethods.cuda_warping_remap(src.CvPtr, dst.CvPtr, map1.CvPtr, map2.CvPtr, (int)interpolation, (int)borderMode
                , borderValue0, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
            GC.KeepAlive(map1);
            GC.KeepAlive(map2);
        }
        #endregion

        #region cuda WarpAffine
        /// <summary>
        /// Applies an affine transformation to an image.
        /// </summary>
        /// <param name="src">Source image. CV_8U , CV_16U , CV_32S , or CV_32F depth and 1, 3, or 4 channels are
        /// supported.</param>
        /// <param name="dst">Destination image with the same type as src . The size is dsize .</param>
        /// <param name="m">*2x3* transformation matrix.</param>
        /// <param name="dsize">Size of the destination image.</param>
        /// <param name="flags">Combination of interpolation methods (see resize) and the optional flag
        /// WARP_INVERSE_MAP specifying that M is an inverse transformation(dst=\>src ). Only
        /// INTER_NEAREST, INTER_LINEAR, and INTER_CUBIC interpolation methods are supported.</param>
        /// <param name="borderMode">pixel extrapolation method; when borderMode=BORDER_TRANSPARENT, 
        /// it means that the pixels in the destination image corresponding to the "outliers" 
        /// in the source image are not modified by the function.</param>
        /// <param name="borderValue">value used in case of a constant border; by default, it is 0.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void warpAffine(
            InputArray src, OutputArray dst, InputArray m, Size dsize,
            InterpolationFlags flags = InterpolationFlags.Linear,
            BorderTypes borderMode = BorderTypes.Constant, Scalar? borderValue = null, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            if (m == null)
                throw new ArgumentNullException(nameof(m));
            src.ThrowIfDisposed();
            dst.ThrowIfDisposed();
            m.ThrowIfDisposed();
            Scalar borderValue0 = borderValue.GetValueOrDefault(Scalar.All(0));
            NativeMethods.cuda_warping_warpAffine(src.CvPtr, dst.CvPtr, m.CvPtr, dsize, (int)flags, (int)borderMode, borderValue0, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            GC.KeepAlive(m);
            dst.Fix();
        }
        #endregion

        #region cuda buildWarpAffineMaps
        /// <summary>
        /// Builds transformation maps for affine transformation.
        /// </summary>
        /// <param name="M">*2x3* transformation matrix.</param>
        /// <param name="inverse">Flag specifying that M is an inverse transformation ( dst=\>src ).</param>
        /// <param name="dsize">Size of the destination image.</param>
        /// <param name="xmap">X values with CV_32FC1 type.</param>
        /// <param name="ymap">Y values with CV_32FC1 type.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void buildWarpAffineMaps(
            InputArray M, bool inverse, Size dsize, OutputArray xmap, OutputArray ymap, Stream stream = null) {
            if (M == null)
                throw new ArgumentNullException(nameof(M));
            if (xmap == null)
                throw new ArgumentNullException(nameof(xmap));
            if (ymap == null)
                throw new ArgumentNullException(nameof(ymap));
            M.ThrowIfDisposed();
            xmap.ThrowIfDisposed();
            ymap.ThrowIfDisposed();
            NativeMethods.cuda_warping_buildWarpAffineMaps(M.CvPtr, inverse, dsize, xmap.CvPtr
                , ymap.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(M);
            GC.KeepAlive(xmap);
            GC.KeepAlive(ymap);
            xmap.Fix();
            ymap.Fix();
        }
        #endregion

        #region cuda warpPerspective
        /// <summary>
        /// Applies a perspective transformation to an image.
        /// </summary>
        /// <param name="src">Source image. CV_8U , CV_16U , CV_32S , or CV_32F depth and 1, 3, or 4 channels are
        /// supported.</param>
        /// <param name="dst">Destination image with the same type as src . The size is dsize .</param>
        /// <param name="m">*3x3* transformation matrix.</param>
        /// <param name="dsize">Size of the destination image.</param>
        /// <param name="flags">Combination of interpolation methods (see resize) and the optional flag
        /// WARP_INVERSE_MAP specifying that M is an inverse transformation(dst=\>src ). Only
        /// INTER_NEAREST, INTER_LINEAR, and INTER_CUBIC interpolation methods are supported.</param>
        /// <param name="borderMode">pixel extrapolation method; when borderMode=BORDER_TRANSPARENT, 
        /// it means that the pixels in the destination image corresponding to the "outliers" 
        /// in the source image are not modified by the function.</param>
        /// <param name="borderValue">value used in case of a constant border; by default, it is 0.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void warpPerspective(
            InputArray src, OutputArray dst, InputArray m, Size dsize,
            InterpolationFlags flags = InterpolationFlags.Linear,
            BorderTypes borderMode = BorderTypes.Constant, Scalar? borderValue = null, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            if (m == null)
                throw new ArgumentNullException(nameof(m));
            src.ThrowIfDisposed();
            dst.ThrowIfDisposed();
            m.ThrowIfDisposed();
            Scalar borderValue0 = borderValue.GetValueOrDefault(Scalar.All(0));
            NativeMethods.cuda_warping_warpPerspective(src.CvPtr, dst.CvPtr, m.CvPtr, dsize, (int)flags, (int)borderMode, borderValue0, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            GC.KeepAlive(m);
            dst.Fix();
        }
        #endregion

        #region cuda buildWarpPerspectiveMaps
        /// <summary>
        /// Builds transformation maps for perspective transformation.
        /// </summary>
        /// <param name="M">*3x3* transformation matrix.</param>
        /// <param name="inverse">Flag specifying that M is an inverse transformation ( dst=\>src ).</param>
        /// <param name="dsize">Size of the destination image.</param>
        /// <param name="xmap">X values with CV_32FC1 type.</param>
        /// <param name="ymap">Y values with CV_32FC1 type.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void buildWarpPerspectiveMaps(
            InputArray M, bool inverse, Size dsize, OutputArray xmap, OutputArray ymap, Stream stream = null) {
            if (M == null)
                throw new ArgumentNullException(nameof(M));
            if (xmap == null)
                throw new ArgumentNullException(nameof(xmap));
            if (ymap == null)
                throw new ArgumentNullException(nameof(ymap));
            M.ThrowIfDisposed();
            xmap.ThrowIfDisposed();
            ymap.ThrowIfDisposed();
            NativeMethods.cuda_warping_buildWarpPerspectiveMaps(M.CvPtr, inverse, dsize, xmap.CvPtr
                , ymap.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(M);
            GC.KeepAlive(xmap);
            GC.KeepAlive(ymap);
            xmap.Fix();
            ymap.Fix();
        }
        #endregion

        #region cuda rotate
        /// <summary>
        /// Rotates an image around the origin (0,0) and then shifts it.
        /// </summary>
        /// <param name="src">Source image. Supports 1, 3 or 4 channels images with CV_8U , CV_16U or CV_32F
        /// depth.</param>
        /// <param name="dst">Destination image with the same type as src . The size is dsize .</param>
        /// <param name="dsize">Size of the destination image.</param>
        /// <param name="angle">Angle of rotation in degrees.</param>
        /// <param name="xShift">Shift along the horizontal axis.</param>
        /// <param name="yShift">Shift along the vertical axis.</param>
        /// <param name="flags">Interpolation method. Only INTER_NEAREST , INTER_LINEAR , and INTER_CUBIC
        /// are supported.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void rotate(
            InputArray src, OutputArray dst, Size dsize, double angle,double xShift = 0,double yShift = 0,
            InterpolationFlags flags = InterpolationFlags.Linear,Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfDisposed();

            NativeMethods.cuda_warping_rotate(src.CvPtr, dst.CvPtr, dsize, angle, xShift, yShift, (int)flags, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda Resize
        /// <summary>
        /// Resizes an image.
        /// </summary>
        /// <param name="src">Source image.</param>
        /// <param name="dst">Destination image with the same type as src . The size is dsize (when it is non-zero)
        /// or the size is computed from src.size() , fx , and fy.</param>
        /// <param name="dsize">Destination image size. If it is zero, it is computed as:
        /// \f[\texttt{dsize = Size(round(fx* src.cols), round(fy* src.rows))}\f]
        /// Either dsize or both fx and fy must be non-zero.</param>
        /// <param name="fx">Scale factor along the horizontal axis. If it is zero, it is computed as:
        /// \f[\texttt{(double) dsize.width/src.cols}\f]</param>
        /// <param name="fy">Scale factor along the vertical axis. If it is zero, it is computed as:
        /// \f[\texttt{(double) dsize.height/src.rows}\f]</param>
        /// <param name="interpolation">interpolation Interpolation method. INTER_NEAREST , INTER_LINEAR and INTER_CUBIC are
        /// supported for now.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void Resize(InputArray src, OutputArray dst, Size dsize,
            double fx = 0, double fy = 0, InterpolationFlags interpolation = InterpolationFlags.Linear, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.cuda_warping_resize(src.CvPtr, dst.CvPtr, dsize, fx, fy, (int)interpolation, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda PyrDown
        /// <summary>
        /// Smoothes an image and downsamples it.
        /// </summary>
        /// <param name="src">Source image.</param>
        /// <param name="dst">Destination image. Will have Size((src.cols+1)/2, (src.rows+1)/2) size and the same
        /// type as src.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void pyrDown(InputArray src, OutputArray dst, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.cuda_warping_pyrDown(src.CvPtr, dst.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda Pyrup
        /// <summary>
        /// Upsamples an image and then smoothes it.
        /// </summary>
        /// <param name="src">Source image.</param>
        /// <param name="dst">Destination image. Will have Size(src.cols\*2, src.rows\*2) size and the same type as
        /// src.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void pyrUp(InputArray src, OutputArray dst, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.cuda_warping_pyrUp(src.CvPtr, dst.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion
    }
}
