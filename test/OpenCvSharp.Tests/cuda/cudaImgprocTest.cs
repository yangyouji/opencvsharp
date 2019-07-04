using OpenCvSharp.Cuda;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace OpenCvSharp.Tests.cuda {
    public class cudaImgprocTest : TestBase {
        public cudaImgprocTest(ITestOutputHelper output) : base(output) {

        }

        [Fact]
        public void cuda_demosaicing() {
            Mat src = Image("lenna.png", ImreadModes.Grayscale);
            Size size = src.Size();

            using (GpuMat g_src = new GpuMat(size, MatType.CV_8UC1))
            using (GpuMat dst = new GpuMat()) {
                g_src.Upload(src);
                Cuda.cuda.demosaicing(g_src, dst,  ColorConversionCodes.BayerBG2BGR);
                ShowImagesWhenDebugMode(g_src, dst);
            }
        }

        [Fact]
        public void cuda_swapChannels() {
            Mat src = Image("lenna.png", ImreadModes.AnyColor);
            Cv2.CvtColor(src, src, ColorConversionCodes.BGR2BGRA);
            Size size = src.Size();

            using (GpuMat g_src = new GpuMat(size, src.Type()))
            using (GpuMat dst = new GpuMat()) {
                g_src.Upload(src);
                int[] dstOrder = { 2, 1, 0, 3 };
                Cuda.cuda.swapChannels(g_src, dstOrder);

                Mat dst_gold = new Mat();
                Cv2.CvtColor(src, dst_gold, ColorConversionCodes.BGRA2RGBA);
                ImageEquals(dst_gold, g_src);
                ShowImagesWhenDebugMode(src, g_src);
            }
        }

        [Fact]
        public void cuda_calcHist() {
            Mat src = Image("lenna.png", ImreadModes.Grayscale);
            Size size = src.Size();

            using (GpuMat g_src = new GpuMat(size, src.Type()))
            using (GpuMat hist = new GpuMat()) {
                g_src.Upload(src);
                Cuda.cuda.calcHist(g_src, hist);

                Mat hist_gold = new Mat();

                Mat[] r_src = { src };
                int hbins = 256;
                float[] hranges = { 0.0f, 256.0f };
                int[] histSize = { hbins };
                float[][] ranges = { hranges };
                int[] channels = { 0 };

                Cv2.CalcHist(r_src, channels, new Mat(), hist_gold, 1, histSize, ranges);
                hist_gold = hist_gold.Reshape(1, 1);
                hist_gold.ConvertTo(hist_gold, MatType.CV_32S);

                ImageEquals(hist_gold, hist);
            }
        }

        [Fact]
        public void cuda_calcHist_mask() {
            Mat src = Image("lenna.png", ImreadModes.Grayscale);
            Mat mask = new Mat(src.Size(), MatType.CV_8UC1,Scalar.White);
            mask.Rectangle(new Rect(0, 0, mask.Width / 2, mask.Height / 2), Scalar.Black, -1);
            Size size = src.Size();

            using (GpuMat g_src = new GpuMat(size, src.Type()))
            using (GpuMat hist = new GpuMat()) {
                g_src.Upload(src);
                GpuMat g_mask = new GpuMat();
                g_mask.Upload(mask);

                Cuda.cuda.calcHist(g_src, g_mask, hist);

                Mat hist_gold = new Mat();

                Mat[] r_src = { src };
                int hbins = 256;
                float[] hranges = { 0.0f, 256.0f };
                int[] histSize = { hbins };
                float[][] ranges = { hranges };
                int[] channels = { 0 };

                Cv2.CalcHist(r_src, channels, mask, hist_gold, 1, histSize, ranges);
                hist_gold = hist_gold.Reshape(1, 1);
                hist_gold.ConvertTo(hist_gold, MatType.CV_32S);

                ImageEquals(hist_gold, hist);
            }
        }

        [Fact]
        public void cuda_equalizeHist() {
            Mat src = Image("lenna.png", ImreadModes.Grayscale);
            Size size = src.Size();

            using (GpuMat g_src = new GpuMat(size, src.Type()))
            using (GpuMat dst = new GpuMat()) {
                g_src.Upload(src);
                Cuda.cuda.equalizeHist(g_src, dst);

                Mat dst_gold = new Mat();
                Cv2.EqualizeHist(src, dst_gold);
                ImageEquals(dst_gold, dst, 3.0);
                ShowImagesWhenDebugMode(src, dst);
            }
        }

        [Fact]
        public void cuda_CLAHE() {
            Mat src = Image("lenna.png", ImreadModes.Grayscale);
            Size size = src.Size();

            Cuda.CLAHE clahe = Cuda.CLAHE.create(20.0);
            CLAHE clahe_gold = CLAHE.Create(20.0);

            using (GpuMat g_src = new GpuMat(size, src.Type()))
            using (GpuMat dst = new GpuMat()) {
                g_src.Upload(src);

                clahe.Apply(g_src, dst);
                Mat dst_gold = new Mat();
                clahe_gold.Apply(src, dst_gold);

                ImageEquals(dst_gold, dst, 1.0);
                ShowImagesWhenDebugMode(src, dst);
            }
        }

        [Fact]
        public void cuda_histEven() {
            Mat src = Image("lenna.png", ImreadModes.Grayscale);
            Size size = src.Size();

            int hbins = 30;
            float[] hranges = { 50.0f, 200.0f };

            using (GpuMat g_src = new GpuMat(size, src.Type()))
            using (GpuMat hist = new GpuMat()) {
                g_src.Upload(src);
                Cuda.cuda.histEven(g_src, hist, hbins, (int)hranges[0], (int)hranges[1]);

                Mat hist_gold = new Mat();
                int[] histSize = { hbins };
                float[][] ranges = { hranges };
                int[] channels = { 0 };
                Mat[] r_src = { src };
                Cv2.CalcHist(r_src, channels, new Mat(), hist_gold, 1, histSize, ranges);

                hist_gold = hist_gold.T();
                hist_gold.ConvertTo(hist_gold, MatType.CV_32S);

                ImageEquals(hist_gold, hist);
            }
        }

        [Fact]
        public void cuda_canny() {
            Mat src = Image("lenna.png", ImreadModes.Grayscale);
            Size size = src.Size();

            double low_thresh = 50.0;
            double high_thresh = 100.0;

            using (GpuMat g_src = new GpuMat(size, src.Type()))
            using (GpuMat edges = new GpuMat()) {
                g_src.Upload(src);

                Cuda.CannyEdgeDetector canny = Cuda.CannyEdgeDetector.create(low_thresh, high_thresh);
                canny.detect(g_src, edges);
                //output.WriteLine("out: {0}", canny.LowThreshold);

                Mat edges_gold = new Mat();
                Cv2.Canny(src, edges_gold, low_thresh, high_thresh);

                ImageEquals(edges_gold, edges, 2);
                ShowImagesWhenDebugMode(edges_gold, edges);
            }
        }

        [Fact]
        public void cuda_HoughLines() {
            Mat src = Mat.Zeros(128,128,MatType.CV_8UC1);
            Size size = src.Size();

            Cv2.Line(src, new Point(20, 0), new Point(20, src.Rows), Scalar.White);
            Cv2.Line(src, new Point(0, 50), new Point(src.Cols, 50), Scalar.White);
            Cv2.Line(src, new Point(0, 0), new Point(src.Cols, src.Rows), Scalar.White);
            Cv2.Line(src, new Point(src.Cols, 0), new Point(0, src.Rows), Scalar.White);

            const float rho = 1.0f;
            const float theta = (float)(1.5 * Cv2.PI / 180.0);
            const int threshold = 100;

            using (GpuMat g_src = new GpuMat(size, src.Type()))
            using (GpuMat d_lines = new GpuMat()) {
                g_src.Upload(src);

                Cuda.HoughLinesDetector houghline = Cuda.HoughLinesDetector.create(rho, theta, threshold);
                houghline.Threshold = threshold;
                houghline.detect(g_src, d_lines);

                LineSegmentPolar[] lines;

                houghline.downloadResults(d_lines, out lines);

                Mat dst = Mat.Zeros(size, MatType.CV_8UC1);
                for (int i = 0; i < lines.Length; i++) {
                    float t_rho = lines[i].Rho;
                    float t_theta = lines[i].Theta;
                    Point pt1, pt2;
                    double a = System.Math.Cos(t_theta), b = System.Math.Sin(t_theta);
                    double x0 = a * t_rho, y0 = b * t_rho;
                    pt1.X = (int)System.Math.Round(x0 + 1000 * (-b));
                    pt1.Y = (int)System.Math.Round(y0 + 1000 * (a));
                    pt2.X = (int)System.Math.Round(x0 - 1000 * (-b));
                    pt2.Y = (int)System.Math.Round(y0 - 1000 * (a));
                    Cv2.Line(dst, pt1, pt2, Scalar.White);
                }

                ImageEquals(src, dst, 2);
                ShowImagesWhenDebugMode(src, dst);
            }
        }

        [Fact]
        public void cuda_HoughSegment() {
            Mat src = Mat.Zeros(128, 128, MatType.CV_8UC1);
            Size size = src.Size();

            Cv2.Line(src, new Point(20, 0), new Point(20, src.Rows), Scalar.White);
            Cv2.Line(src, new Point(0, 50), new Point(src.Cols, 50), Scalar.White);
            Cv2.Line(src, new Point(0, 0), new Point(src.Cols, src.Rows), Scalar.White);
            Cv2.Line(src, new Point(src.Cols, 0), new Point(0, src.Rows), Scalar.White);

            const float rho = 1.0f;
            const float theta = (float)(1.5 * Cv2.PI / 180.0);
            const int minilinelenth = 10;
            const int maxlinegap = 2;

            using (GpuMat g_src = new GpuMat(size, src.Type()))
            using (GpuMat d_lines = new GpuMat()) {
                g_src.Upload(src);

                Cuda.HoughSegmentDetector houghline = Cuda.HoughSegmentDetector.create(rho, theta, minilinelenth,maxlinegap);
                houghline.detect(g_src, d_lines);

                LineSegmentPoint[] lines;
               
                houghline.downloadResults(d_lines, out lines);

                Mat dst = Mat.Zeros(size, MatType.CV_8UC1);
                for (int i = 0; i < lines.Length; i++) {
                    Cv2.Line(dst, lines[i].P1, lines[i].P2, Scalar.White);
                }

                ImageEquals(src, dst, 2);
                ShowImagesWhenDebugMode(src, g_src);
            }
        }

        [Fact]
        public void cuda_HoughCircles() {
            Mat src = Mat.Zeros(128, 128, MatType.CV_8UC1);
            Size size = src.Size();

            const float dp = 2.0f;
            const float minDist = 0.0f;
            const int minRadius = 10;
            const int maxRadius = 20;
            const int cannyThreshold = 100;
            const int votesThreshold = 20;

            List<Vec3f> circles_gold = new List<Vec3f>();
            circles_gold.Add(new Vec3f(20, 20, minRadius));
            circles_gold.Add(new Vec3f(90, 87, minRadius + 3));
            circles_gold.Add(new Vec3f(30, 70, minRadius + 8));
            circles_gold.Add(new Vec3f(80, 10, maxRadius));


            for (int i = 0; i < circles_gold.Count; ++i) {
                Cv2.Circle(src, new Point(circles_gold[i][0], circles_gold[i][1]), (int)circles_gold[i][2], Scalar.White,-1);
            }
            using (GpuMat g_src = new GpuMat(size, src.Type()))
            using (GpuMat d_circles = new GpuMat()) {
                g_src.Upload(src);

                HoughCirclesDetector houghCircles = HoughCirclesDetector.create(dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius);

                houghCircles.detect(g_src, d_circles);



                CircleSegment[] circles;

                houghCircles.downloadResults(d_circles, out circles);

                for (int i = 0; i < circles.Length; ++i) {
                    bool found = false;

                    for (int j = 0; j < circles_gold.Count; ++j) {
                        Vec3f gold = circles_gold[j];

                        if (System.Math.Abs(circles[i].Center.X - gold[0]) < 5 && System.Math.Abs(circles[i].Center.Y - gold[1]) < 5 && System.Math.Abs(circles[i].Radius - gold[2]) < 5) {
                            found = true;
                            break;
                        }
                    }

                    Assert.True(found);
                }
                ShowImagesWhenDebugMode(src, g_src);
            }
        }

    }
}
