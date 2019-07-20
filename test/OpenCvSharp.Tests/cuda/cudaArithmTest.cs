using OpenCvSharp.Cuda;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace OpenCvSharp.Tests.cuda {
    public class cudaArithmTest : TestBase {
        public cudaArithmTest(ITestOutputHelper output) : base(output) {

        }

        [Fact]
        public void cuda_add() {
            Mat mat1 = Image("lenna.png", ImreadModes.Grayscale);
            Size size = mat1.Size();
            Mat mat2 = new Mat(size, mat1.Type(), Scalar.Black);

            using (GpuMat g_mat1 = new GpuMat(size, mat1.Type()))
            using (GpuMat dst = new GpuMat()) {
                GpuMat g_mat2 = new GpuMat(size, mat2.Type());
                g_mat2.Upload(mat2);
                g_mat1.Upload(mat1);

                Cuda.cuda.add(g_mat1, g_mat2, dst);

                Mat dst_gold = new Mat(size, mat1.Type(), Scalar.Black);
                Cv2.Add(mat1, mat2, dst_gold);
                ImageEquals(dst_gold, dst);
                ShowImagesWhenDebugMode(g_mat1, dst);
            }
        }

        [Fact]
        public void cuda_subtract() {
            Mat mat1 = Image("lenna.png", ImreadModes.Grayscale);
            Size size = mat1.Size();
            Mat mat2 = new Mat(size, mat1.Type(), new Scalar(2));

            using (GpuMat g_mat1 = new GpuMat(size, mat1.Type()))
            using (GpuMat dst = new GpuMat()) {
                GpuMat g_mat2 = new GpuMat(size, mat2.Type());
                g_mat2.Upload(mat2);
                g_mat1.Upload(mat1);

                Cuda.cuda.subtract(g_mat1, g_mat2, dst);

                Mat dst_gold = new Mat(size, mat1.Type(), Scalar.Black);
                Cv2.Subtract(mat1, mat2, dst_gold);
                ImageEquals(dst_gold, dst);
                ShowImagesWhenDebugMode(g_mat1, dst);
            }
        }

        [Fact]
        public void cuda_multiply() {
            Mat mat1 = Image("lenna.png", ImreadModes.Grayscale);
            Size size = mat1.Size();
            Mat mat2 = new Mat(size, mat1.Type(), new Scalar(2));

            using (GpuMat g_mat1 = new GpuMat(size, mat1.Type()))
            using (GpuMat dst = new GpuMat()) {
                GpuMat g_mat2 = new GpuMat(size, mat2.Type());
                g_mat2.Upload(mat2);
                g_mat1.Upload(mat1);

                Cuda.cuda.multiply(g_mat1, g_mat2, dst, 5.3);

                Mat dst_gold = new Mat(size, mat1.Type(), Scalar.Black);
                Cv2.Multiply(mat1, mat2, dst_gold, 5.3);
                ImageEquals(dst_gold, dst);
                ShowImagesWhenDebugMode(g_mat1, dst);
            }
        }

        [Fact]
        public void cuda_divide() {
            Mat mat1 = Image("lenna.png", ImreadModes.Grayscale);
            Size size = mat1.Size();
            Mat mat2 = new Mat(size, mat1.Type(), new Scalar(2));

            using (GpuMat g_mat1 = new GpuMat(size, mat1.Type()))
            using (GpuMat dst = new GpuMat()) {
                GpuMat g_mat2 = new GpuMat(size, mat2.Type());
                g_mat2.Upload(mat2);
                g_mat1.Upload(mat1);

                Cuda.cuda.divide(g_mat1, g_mat2, dst, 5.3);

                Mat dst_gold = new Mat(size, mat1.Type(), Scalar.Black);
                Cv2.Divide(mat1, mat2, dst_gold, 5.3);
                ImageEquals(dst_gold, dst);
                ShowImagesWhenDebugMode(g_mat1, dst);
            }
        }

        [Fact]
        public void cuda_absdiff() {
            Mat mat1 = Image("lenna.png", ImreadModes.Grayscale);
            Size size = mat1.Size();
            Mat mat2 = new Mat(size, mat1.Type(), new Scalar(2));

            using (GpuMat g_mat1 = new GpuMat(size, mat1.Type()))
            using (GpuMat dst = new GpuMat()) {
                GpuMat g_mat2 = new GpuMat(size, mat2.Type());
                g_mat2.Upload(mat2);
                g_mat1.Upload(mat1);

                Cuda.cuda.absdiff(g_mat1, g_mat2, dst);

                Mat dst_gold = new Mat(size, mat1.Type(), Scalar.Black);
                Cv2.Absdiff(mat1, mat2, dst_gold);
                ImageEquals(dst_gold, dst);
                ShowImagesWhenDebugMode(g_mat1, dst);
            }
        }

        [Fact]
        public void cuda_abs() {
            Mat mat1 = Image("lenna.png", ImreadModes.Grayscale);
            Size size = mat1.Size();

            using (GpuMat g_mat1 = new GpuMat(size, mat1.Type()))
            using (GpuMat dst = new GpuMat()) {
                g_mat1.Upload(mat1);

                Cuda.cuda.abs(g_mat1, dst);

                Mat dst_gold = Cv2.Abs(mat1);
                ImageEquals(dst_gold, dst);
                ShowImagesWhenDebugMode(g_mat1, dst);
            }
        }

        [Fact]
        public void cuda_sqr() {
            Mat mat1 = Image("lenna.png", ImreadModes.Grayscale);
            Size size = mat1.Size();

            using (GpuMat g_mat1 = new GpuMat(size, mat1.Type()))
            using (GpuMat dst = new GpuMat()) {
                g_mat1.Upload(mat1);

                Cuda.cuda.sqr(g_mat1, dst);

                Mat dst_gold = new Mat();
                Cv2.Multiply(mat1, mat1, dst_gold);
                ImageEquals(dst_gold, dst);
                ShowImagesWhenDebugMode(g_mat1, dst);
            }
        }

        [Fact]
        public void cuda_sqrt() {
            Mat mat1 = Image("lenna.png", ImreadModes.Grayscale);
            mat1.ConvertTo(mat1, MatType.CV_32FC1);
            Size size = mat1.Size();

            using (GpuMat g_mat1 = new GpuMat(size, mat1.Type()))
            using (GpuMat dst = new GpuMat()) {
                g_mat1.Upload(mat1);

                Cuda.cuda.sqrt(g_mat1, dst);

                Mat dst_gold = new Mat();
                Cv2.Sqrt(mat1, dst_gold);
                ImageEquals(dst_gold, dst);
                ShowImagesWhenDebugMode(g_mat1, dst);
            }
        }

        [Fact]
        public void cuda_log() {
            Mat mat1 = Image("lenna.png", ImreadModes.Grayscale);
            mat1.ConvertTo(mat1, MatType.CV_32FC1);
            Size size = mat1.Size();

            using (GpuMat g_mat1 = new GpuMat(size, mat1.Type()))
            using (GpuMat dst = new GpuMat()) {
                g_mat1.Upload(mat1);

                Cuda.cuda.log(g_mat1, dst);

                Mat dst_gold = new Mat();
                Cv2.Log(mat1, dst_gold);
                ImageEquals(dst_gold, dst, 1e-6);
                ShowImagesWhenDebugMode(g_mat1, dst);
            }
        }

        [Fact]
        public void cuda_exps() {
            Mat mat1 = Image("lenna.png", ImreadModes.Grayscale);
            mat1.ConvertTo(mat1, MatType.CV_32FC1);
            Size size = mat1.Size();

            using (GpuMat g_mat1 = new GpuMat(size, mat1.Type()))
            using (GpuMat dst = new GpuMat()) {
                g_mat1.Upload(mat1);

                Cuda.cuda.exp(g_mat1, dst);

                Mat dst_gold = new Mat();
                Cv2.Exp(mat1, dst_gold);
                ImageEquals(dst_gold, dst, 1e-6);
                ShowImagesWhenDebugMode(g_mat1, dst);
            }
        }

        [Fact]
        public void cuda_pow() {
            Mat mat1 = Image("lenna.png", ImreadModes.Grayscale);
            mat1.ConvertTo(mat1, MatType.CV_32FC1);
            Size size = mat1.Size();

            using (GpuMat g_mat1 = new GpuMat(size, mat1.Type()))
            using (GpuMat dst = new GpuMat()) {
                g_mat1.Upload(mat1);

                Cuda.cuda.pow(g_mat1, 2.0, dst);

                Mat dst_gold = new Mat();
                Cv2.Pow(mat1, 2.0, dst_gold);
                ImageEquals(dst_gold, dst, 1e-1);
                ShowImagesWhenDebugMode(g_mat1, dst);
            }
        }

        [Fact]
        public void cuda_compare() {
            Mat mat1 = Image("lenna.png", ImreadModes.Grayscale);
            Size size = mat1.Size();
            Mat mat2 = new Mat(size, mat1.Type(), new Scalar(2));

            using (GpuMat g_mat1 = new GpuMat(size, mat1.Type()))
            using (GpuMat dst = new GpuMat()) {
                GpuMat g_mat2 = new GpuMat(size, mat2.Type());
                g_mat2.Upload(mat2);
                g_mat1.Upload(mat1);

                Cuda.cuda.compare(g_mat1, g_mat2, dst, CmpTypes.EQ);

                Mat dst_gold = new Mat(size, mat1.Type(), Scalar.Black);
                Cv2.Compare(mat1, mat2, dst_gold, CmpTypes.EQ);
                ImageEquals(dst_gold, dst);
                ShowImagesWhenDebugMode(g_mat1, dst);
            }
        }

        [Fact]
        public void cuda_bitwise_not() {
            Mat mat1 = Image("lenna.png", ImreadModes.Grayscale);
            Size size = mat1.Size();

            using (GpuMat g_mat1 = new GpuMat(size, mat1.Type()))
            using (GpuMat dst = new GpuMat()) {
                g_mat1.Upload(mat1);

                Cuda.cuda.bitwise_not(g_mat1, dst);

                Mat dst_gold = new Mat();
                dst_gold = ~mat1;
                ImageEquals(dst_gold, dst);
                ShowImagesWhenDebugMode(g_mat1, dst);
            }
        }

        [Fact]
        public void cuda_bitwise_and() {
            Mat mat1 = Image("lenna.png", ImreadModes.Grayscale);
            Size size = mat1.Size();
            Mat mat2 = new Mat(size, mat1.Type(), new Scalar(2));

            using (GpuMat g_mat1 = new GpuMat(size, mat1.Type()))
            using (GpuMat dst = new GpuMat()) {
                GpuMat g_mat2 = new GpuMat(size, mat2.Type());
                g_mat2.Upload(mat2);
                g_mat1.Upload(mat1);

                Cuda.cuda.bitwise_and(g_mat1, g_mat2, dst);

                Mat dst_gold = new Mat(size, mat1.Type(), Scalar.Black);
                dst_gold = mat1 & mat2;
                ImageEquals(dst_gold, dst);
                ShowImagesWhenDebugMode(g_mat1, dst);
            }
        }

        [Fact]
        public void cuda_bitwise_xor() {
            Mat mat1 = Image("lenna.png", ImreadModes.Grayscale);
            Size size = mat1.Size();
            Mat mat2 = new Mat(size, mat1.Type(), new Scalar(2));

            using (GpuMat g_mat1 = new GpuMat(size, mat1.Type()))
            using (GpuMat dst = new GpuMat()) {
                GpuMat g_mat2 = new GpuMat(size, mat2.Type());
                g_mat2.Upload(mat2);
                g_mat1.Upload(mat1);

                Cuda.cuda.bitwise_xor(g_mat1, g_mat2, dst);

                Mat dst_gold = new Mat(size, mat1.Type(), Scalar.Black);
                dst_gold = mat1 ^ mat2;
                ImageEquals(dst_gold, dst);
                ShowImagesWhenDebugMode(g_mat1, dst);
            }
        }

        [Fact]
        public void cuda_min() {
            Mat mat1 = Image("lenna.png", ImreadModes.Grayscale);
            Size size = mat1.Size();
            Mat mat2 = new Mat(size, mat1.Type(), new Scalar(2));

            using (GpuMat g_mat1 = new GpuMat(size, mat1.Type()))
            using (GpuMat dst = new GpuMat()) {
                GpuMat g_mat2 = new GpuMat(size, mat2.Type());
                g_mat2.Upload(mat2);
                g_mat1.Upload(mat1);

                Cuda.cuda.min(g_mat1, g_mat2, dst);

                Mat dst_gold = new Mat(size, mat1.Type(), Scalar.Black);
                Cv2.Min(mat1, mat2, dst_gold);
                ImageEquals(dst_gold, dst);
                ShowImagesWhenDebugMode(g_mat1, dst);
            }
        }

        [Fact]
        public void cuda_max() {
            Mat mat1 = Image("lenna.png", ImreadModes.Grayscale);
            Size size = mat1.Size();
            Mat mat2 = new Mat(size, mat1.Type(), new Scalar(2));

            using (GpuMat g_mat1 = new GpuMat(size, mat1.Type()))
            using (GpuMat dst = new GpuMat()) {
                GpuMat g_mat2 = new GpuMat(size, mat2.Type());
                g_mat2.Upload(mat2);
                g_mat1.Upload(mat1);

                Cuda.cuda.max(g_mat1, g_mat2, dst);

                Mat dst_gold = new Mat(size, mat1.Type(), Scalar.Black);
                Cv2.Max(mat1, mat2, dst_gold);
                ImageEquals(dst_gold, dst);
                ShowImagesWhenDebugMode(g_mat1, dst);
            }
        }

        [Fact]
        public void cuda_addWeighted() {
            Mat mat1 = Image("lenna.png", ImreadModes.Grayscale);
            Size size = mat1.Size();
            Mat mat2 = new Mat(size, mat1.Type(), new Scalar(2));
            double alpha = 0.9;
            double beta = 1.1;
            double gamma = 2.3;

            using (GpuMat g_mat1 = new GpuMat(size, mat1.Type()))
            using (GpuMat dst = new GpuMat()) {
                GpuMat g_mat2 = new GpuMat(size, mat2.Type());
                g_mat2.Upload(mat2);
                g_mat1.Upload(mat1);

                Cuda.cuda.addWeighted(g_mat1, alpha, g_mat2, beta, gamma, dst);

                Mat dst_gold = new Mat(size, mat1.Type(), Scalar.Black);
                Cv2.AddWeighted(mat1, alpha, mat2, beta, gamma, dst_gold);
                ImageEquals(dst_gold, dst, 2.0);
                ShowImagesWhenDebugMode(g_mat1, dst);
            }
        }

        [Fact]
        public void cuda_threshold() {
            Mat src = Image("lenna.png", ImreadModes.Grayscale);
            Size size = src.Size();

            double thresh = 127;
            double maxVal = 255;

            using (GpuMat g_src = new GpuMat(size, src.Type()))
            using (GpuMat dst = new GpuMat()) {
                g_src.Upload(src);

                Cuda.cuda.threshold(g_src, dst, thresh, maxVal,ThresholdTypes.Binary);

                Mat dst_gold = new Mat(size, src.Type(), Scalar.Black);
                Cv2.Threshold(src, dst_gold, thresh, maxVal, ThresholdTypes.Binary);
                ImageEquals(dst_gold, dst);
                ShowImagesWhenDebugMode(g_src, dst);
            }
        }

        [Fact]
        public void cuda_magnitude() {
            Mat x = Image("lenna.png", ImreadModes.Grayscale);
            x.ConvertTo(x, MatType.CV_32FC1);
            Mat y = new Mat();
            x.CopyTo(y);
            Size size = x.Size();

            using (GpuMat g_x = new GpuMat(size, x.Type()))
            using (GpuMat dst = new GpuMat()) {
                g_x.Upload(x);
                GpuMat g_y = new GpuMat(size, y.Type());
                g_y.Upload(y);

                Cuda.cuda.magnitude(g_x, g_y, dst);

                Mat dst_gold = new Mat(size, x.Type(), Scalar.Black);
                Cv2.Magnitude(x,y, dst_gold);
                ImageEquals(dst_gold, dst, 1e-4);
                ShowImagesWhenDebugMode(g_x, dst);
            }
        }

        [Fact]
        public void cuda_magnitudeSqr() {
            Mat x = Image("lenna.png", ImreadModes.Grayscale);
            x.ConvertTo(x, MatType.CV_32FC1);
            Mat y = new Mat();
            x.CopyTo(y);
            Size size = x.Size();

            using (GpuMat g_x = new GpuMat(size, x.Type()))
            using (GpuMat dst = new GpuMat()) {
                g_x.Upload(x);
                GpuMat g_y = new GpuMat(size, y.Type());
                g_y.Upload(y);

                Cuda.cuda.magnitudeSqr(g_x, g_y, dst);

                Mat dst_gold = new Mat(size, x.Type(), Scalar.Black);
                Cv2.Magnitude(x, y, dst_gold);
                Cv2.Multiply(dst_gold, dst_gold, dst_gold);

                ImageEquals(dst_gold, dst, 1e-1);
                ShowImagesWhenDebugMode(g_x, dst);
            }
        }

    }
}
