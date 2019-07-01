using OpenCvSharp.Cuda;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Xunit;

namespace OpenCvSharp.Tests.cuda {
    public class cudaWarpingTest : TestBase {
        [Fact]
        public void cuda_remap() {
            Mat src = Image("lenna.png", ImreadModes.Grayscale);
            Size size = src.Size();

            double aplha = Cv2.PI / 4;
            double[,] M = new double[2, 3]  { {System.Math.Cos(aplha), -System.Math.Sin(aplha), size.Width / 2.0},
                                  {System.Math.Sin(aplha),  System.Math.Cos(aplha), 0.0}};
            Mat xmap = new Mat(size, MatType.CV_32FC1);
            Mat ymap = new Mat(size, MatType.CV_32FC1);

            for (int y = 0; y < size.Height; ++y) {
                for (int x = 0; x < size.Width; ++x) {
                    xmap.Set<float>(y, x, (float)(M[0, 0] * x + M[0, 1] * y + M[0, 2]));
                    ymap.Set<float>(y, x, (float)(M[1, 0] * x + M[1, 1] * y + M[1, 2]));
                }
            }

            using (GpuMat g_src = new GpuMat(size, MatType.CV_8UC1))
            using (GpuMat dst = new GpuMat()) {
                GpuMat g_xmap = new GpuMat();
                GpuMat g_ymap = new GpuMat();
                g_src.Upload(src);
                g_xmap.Upload(xmap);
                g_ymap.Upload(ymap);

                Cuda.cuda.remap(g_src, dst, g_xmap, g_ymap);
                ShowImagesWhenDebugMode(src, dst);
            }
        }

        [Fact]
        public void cuda_warpAffine() {
            Mat src = Image("lenna.png", ImreadModes.Grayscale);

            Size srcSize = src.Size();
            double angle = Cv2.PI / 4;

            Mat M = new Mat(2, 3, MatType.CV_64FC1);

            M.Set<double>(0, 0, System.Math.Cos(angle));
            M.Set<double>(0, 1, -System.Math.Sin(angle));
            M.Set<double>(0, 2, srcSize.Width / 2);
            M.Set<double>(1, 0, System.Math.Sin(angle));
            M.Set<double>(1, 1, System.Math.Cos(angle));
            M.Set<double>(1, 2, 0.0);

            using (GpuMat g_src = new GpuMat(srcSize, MatType.CV_8UC1))
            using (GpuMat dst = new GpuMat()) {
                g_src.Upload(src);
                Cuda.cuda.warpAffine(g_src, dst, M, srcSize);
                ShowImagesWhenDebugMode(g_src, dst);
            }
        }

        [Fact]
        public void cuda_buildWarpAffineMaps() {
            Mat src = Image("lenna.png", ImreadModes.Grayscale);

            Size srcSize = src.Size();
            double angle = Cv2.PI / 4;


            Mat M = new Mat(2, 3, MatType.CV_64FC1);

            M.Set<double>(0, 0, System.Math.Cos(angle));
            M.Set<double>(0, 1, -System.Math.Sin(angle));
            M.Set<double>(0, 2, srcSize.Width / 2.0);
            M.Set<double>(1, 0, System.Math.Sin(angle));
            M.Set<double>(1, 1, System.Math.Cos(angle));
            M.Set<double>(1, 2, 0.0);

            using (Mat dst = new Mat())
            using (Mat dst_gold = new Mat()) {
                GpuMat g_xmap = new GpuMat();
                GpuMat g_ymap = new GpuMat();

                Cuda.cuda.buildWarpAffineMaps(M, false, srcSize, g_xmap, g_ymap);


                Cv2.WarpAffine(src, dst_gold, M, srcSize, InterpolationFlags.Nearest, BorderTypes.Constant);

                Mat xmap = new Mat();
                Mat ymap = new Mat();
                g_xmap.Download(xmap);
                g_ymap.Download(ymap);
                Cv2.Remap(src, dst, xmap, ymap, InterpolationFlags.Nearest, BorderTypes.Constant);

                ShowImagesWhenDebugMode(dst_gold, dst);
                //ImageEquals(dst_gold, dst);
            }

        }

        [Fact]
        public void warpPerspective() {
            Mat src = Image("lenna.png", ImreadModes.Grayscale);

            Size srcSize = src.Size();
            double angle = Cv2.PI / 4;

            Mat M = new Mat(3, 3, MatType.CV_64FC1);

            M.Set<double>(0, 0, System.Math.Cos(angle));
            M.Set<double>(0, 1, -System.Math.Sin(angle));
            M.Set<double>(0, 2, srcSize.Width / 2);
            M.Set<double>(1, 0, System.Math.Sin(angle));
            M.Set<double>(1, 1, System.Math.Cos(angle));
            M.Set<double>(1, 2, 0.0);
            M.Set<double>(2, 0, 0.0);
            M.Set<double>(2, 1, 0.0);
            M.Set<double>(2, 2, 1.0);

            using (GpuMat g_src = new GpuMat(srcSize, MatType.CV_8UC1))
            using (GpuMat dst = new GpuMat()) {
                g_src.Upload(src);
                Cuda.cuda.warpPerspective(g_src, dst, M, srcSize);
                ShowImagesWhenDebugMode(g_src, dst);
            }
        }

        [Fact]
        public void cuda_buildWarpPerspectiveMaps() {
            Mat src = Image("lenna.png", ImreadModes.Grayscale);

            Size srcSize = src.Size();
            double angle = Cv2.PI / 4;


            Mat M = new Mat(3, 3, MatType.CV_64FC1);

            M.Set<double>(0, 0, System.Math.Cos(angle));
            M.Set<double>(0, 1, -System.Math.Sin(angle));
            M.Set<double>(0, 2, srcSize.Width / 2.0);
            M.Set<double>(1, 0, System.Math.Sin(angle));
            M.Set<double>(1, 1, System.Math.Cos(angle));
            M.Set<double>(1, 2, 0.0);
            M.Set<double>(2, 0, 0.0);
            M.Set<double>(2, 1, 0.0);
            M.Set<double>(2, 2, 1.0);

            using (Mat dst = new Mat())
            using (Mat dst_gold = new Mat()) {
                GpuMat g_xmap = new GpuMat();
                GpuMat g_ymap = new GpuMat();

                Cuda.cuda.buildWarpPerspectiveMaps(M, false, srcSize, g_xmap, g_ymap);


                Cv2.WarpPerspective(src, dst_gold, M, srcSize, InterpolationFlags.Nearest, BorderTypes.Constant);

                Mat xmap = new Mat();
                Mat ymap = new Mat();
                g_xmap.Download(xmap);
                g_ymap.Download(ymap);
                Cv2.Remap(src, dst, xmap, ymap, InterpolationFlags.Nearest, BorderTypes.Constant);

                ShowImagesWhenDebugMode(dst_gold, dst);
                ImageEquals(dst_gold, dst);
            }

        }

        [Fact]
        public void rotate() {
            Mat src = Image("lenna.png", ImreadModes.Grayscale);

            Size srcSize = src.Size();
            double angle = 45;


            using (GpuMat g_src = new GpuMat(srcSize, MatType.CV_8UC1))
            using (GpuMat dst = new GpuMat()) {
                g_src.Upload(src);
                Cuda.cuda.rotate(g_src, dst, srcSize,angle);
                ShowImagesWhenDebugMode(g_src, dst);
            }
        }

        [Fact]
        public void cuda_pyrdown_pyrup() {
            using (GpuMat src = new GpuMat(100, 100, MatType.CV_8UC1))
            using (GpuMat dst = new GpuMat()) {
                src.SetTo(Scalar.Black);
                Cuda.cuda.pyrDown(src, src);
                Cuda.cuda.pyrUp(src, dst);

                ShowImagesWhenDebugMode(src, dst);

                Assert.Equal(0, Cuda.cuda.CountNonZero(dst));
            }
        }
    }
}
