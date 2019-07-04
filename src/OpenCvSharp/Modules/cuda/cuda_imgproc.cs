using OpenCvSharp.Util;
using System;
using System.Collections.Generic;
using System.Text;

namespace OpenCvSharp.Cuda {

    /// <summary>
    ///  GPU imgproc
    /// </summary>
    public static partial class cuda {

        #region cuda demosaicing
        /// <summary>
        /// Converts an image from Bayer pattern to RGB or grayscale.
        /// </summary>
        /// <param name="src">Source image (8-bit or 16-bit single channel).</param>
        /// <param name="dst">Destination image.</param>
        /// <param name="code">Color space conversion code (see the description below).
        /// The function can do the following transformations:
        /// - Demosaicing using bilinear interpolation
        /// > -   COLOR_BayerBG2GRAY , COLOR_BayerGB2GRAY , COLOR_BayerRG2GRAY , COLOR_BayerGR2GRAY
        /// > -   COLOR_BayerBG2BGR , COLOR_BayerGB2BGR , COLOR_BayerRG2BGR , COLOR_BayerGR2BGR
        /// -   Demosaicing using Malvar-He-Cutler algorithm(@cite MHT2011)
        /// > -   COLOR_BayerBG2GRAY_MHT , COLOR_BayerGB2GRAY_MHT , COLOR_BayerRG2GRAY_MHT ,
        /// >     COLOR_BayerGR2GRAY_MHT
        /// > -   COLOR_BayerBG2BGR_MHT , COLOR_BayerGB2BGR_MHT , COLOR_BayerRG2BGR_MHT ,
        /// >     COLOR_BayerGR2BGR_MHT</param>
        /// <param name="dcn">Number of channels in the destination image. If the parameter is 0, the number of the
        /// channels is derived automatically from src and the code.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void demosaicing(
            InputArray src, OutputArray dst, ColorConversionCodes code, int dcn = -1, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_imgproc_demosaicing(src.CvPtr, dst.CvPtr, (int)code, dcn, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda swapChannels
        /// <summary>
        /// Exchanges the color channels of an image in-place.
        /// </summary>
        /// <param name="image">Source image. Supports only CV_8UC4 type.</param>
        /// <param name="dstOrder">Integer array describing how channel values are permutated. The n-th entry of the
        /// array contains the number of the channel that is stored in the n-th channel of the output image.
        /// E.g.Given an RGBA image, aDstOrder = [3, 2, 1, 0] converts this to ABGR channel order.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        /// <returns></returns>
        public static void swapChannels(InputOutputArray image, IEnumerable<int> dstOrder, Stream stream = null) {
            if (image == null)
                throw new ArgumentNullException(nameof(image));

            int[] dstOrderArray = EnumerableEx.ToArray(dstOrder);
            
            NativeMethods.cuda_imgproc_swapChannels(image.CvPtr, dstOrderArray, stream?.CvPtr ?? Stream.Null.CvPtr);
        }
        #endregion

        #region cuda gammaCorrection
        /// <summary>
        /// Routines for correcting image color gamma.
        /// </summary>
        /// <param name="src">Source image (3- or 4-channel 8 bit).</param>
        /// <param name="dst">Destination image.</param>
        /// <param name="forward">true for forward gamma correction or false for inverse gamma correction.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void gammaCorrection(
            InputArray src, OutputArray dst, bool forward = true, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_imgproc_gammaCorrection(src.CvPtr, dst.CvPtr, forward, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda CalcHist
        /// <summary>
        /// Calculates histogram for one channel 8-bit image.
        /// </summary>
        /// <param name="src">Source image with CV_8UC1 type.</param>
        /// <param name="hist">Destination histogram with one row, 256 columns, and the CV_32SC1 type.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void calcHist(InputArray src, OutputArray hist, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (hist == null)
                throw new ArgumentNullException(nameof(hist));

            NativeMethods.cuda_imgproc_calcHist_0(src.CvPtr, hist.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            hist.Fix();
        }

        /// <summary>
        /// Calculates histogram for one channel 8-bit image confined in given mask.
        /// </summary>
        /// <param name="src">Source image with CV_8UC1 type.</param>
        /// <param name="mask">A mask image same size as src and of type CV_8UC1.</param>
        /// <param name="hist">Destination histogram with one row, 256 columns, and the CV_32SC1 type.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void calcHist(InputArray src, InputArray mask, OutputArray hist, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (mask == null)
                throw new ArgumentNullException(nameof(mask));
            if (hist == null)
                throw new ArgumentNullException(nameof(hist));

            NativeMethods.cuda_imgproc_calcHist_1(src.CvPtr, mask.CvPtr, hist.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(mask);
            hist.Fix();
        }
        #endregion

        #region cuda EqualizeHist
        /// <summary>
        /// normalizes the grayscale image brightness and contrast by normalizing its histogram
        /// </summary>
        /// <param name="src">The source 8-bit single channel image</param>
        /// <param name="dst">The destination image; will have the same size and the same type as src</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void equalizeHist(InputArray src, OutputArray dst, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dst == null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            NativeMethods.cuda_imgproc_equalizeHist(src.CvPtr, dst.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        #endregion

        #region cuda evenLevels
        /// <summary>
        /// Computes levels with even distribution.
        /// </summary>
        /// <param name="levels">Destination array. levels has 1 row, nLevels columns, and the CV_32SC1 type.</param>
        /// <param name="nLevels">Number of computed levels. nLevels must be at least 2.</param>
        /// <param name="lowerLevel">Lower boundary value of the lowest level.</param>
        /// <param name="upperLevel">Upper boundary value of the greatest level.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void evenLevels(OutputArray levels, int nLevels, int lowerLevel, int upperLevel, Stream stream = null) {
            if (levels == null)
                throw new ArgumentNullException(nameof(levels));
            levels.ThrowIfDisposed();
            NativeMethods.cuda_imgproc_evenLevels(levels.CvPtr, nLevels, lowerLevel, upperLevel, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(levels);
            levels.Fix();
        }
        #endregion

        #region cuda histEven
        /// <summary>
        /// Calculates a histogram with evenly distributed bins.
        /// </summary>
        /// <param name="src">Source image. CV_8U, CV_16U, or CV_16S depth and 1 or 4 channels are supported. For
        /// a four-channel image, all channels are processed separately.</param>
        /// <param name="hist">Destination histogram with one row, histSize columns, and the CV_32S type.</param>
        /// <param name="histSize">Size of the histogram.</param>
        /// <param name="lowerLevel">Lower boundary of lowest-level bin.</param>
        /// <param name="upperLevel">Upper boundary of highest-level bin.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void histEven(InputArray src, OutputArray hist, int histSize, int lowerLevel, int upperLevel, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (hist == null)
                throw new ArgumentNullException(nameof(hist));
            src.ThrowIfDisposed();
            hist.ThrowIfNotReady();
            NativeMethods.cuda_imgproc_histEven_0(src.CvPtr, hist.CvPtr, histSize, lowerLevel, upperLevel, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(hist);
            hist.Fix();
        }
        #endregion

        #region cuda histRange
        /// <summary>
        /// Calculates a histogram with evenly distributed bins.
        /// </summary>
        /// <param name="src">Source image. CV_8U, CV_16U, or CV_16S depth and 1 or 4 channels are supported. For
        /// a four-channel image, all channels are processed separately.</param>
        /// <param name="hist">Destination histogram with one row, histSize columns, and the CV_32S type.</param>
        /// <param name="levels">Number of levels in the histogram.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void histRange(InputArray src, OutputArray hist, InputArray levels, Stream stream = null) {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (hist == null)
                throw new ArgumentNullException(nameof(hist));
            if (levels == null)
                throw new ArgumentNullException(nameof(levels));
            src.ThrowIfDisposed();
            hist.ThrowIfNotReady();
            levels.ThrowIfDisposed();
            NativeMethods.cuda_imgproc_histRange_0(src.CvPtr, hist.CvPtr, levels.CvPtr, stream?.CvPtr ?? Stream.Null.CvPtr);
            GC.KeepAlive(src);
            GC.KeepAlive(hist);
            GC.KeepAlive(levels);
            hist.Fix();
        }
        #endregion

    }
}

