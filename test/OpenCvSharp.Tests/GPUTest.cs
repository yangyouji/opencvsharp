using System;
using Xunit;
using Xunit.Abstractions;

namespace OpenCvSharp.Tests {
    // ReSharper disable InconsistentNaming

    public class GPUTest : TestBase {
        public GPUTest(ITestOutputHelper output) : base(output) {

        }

        [Fact]
        public void SimpleGPUTest() {
            Cuda.GpuMat gpumat = new Cuda.GpuMat();
            Cuda.GpuMat gpumat_2 = new Cuda.GpuMat();
            Mat src = Image("lenna.png", ImreadModes.Grayscale);
            gpumat.Upload(src);

            output.WriteLine("out: {0}", gpumat.Cols);
            Cuda.cuda.Resize(gpumat, gpumat, new Size(120, 120));
            output.WriteLine("out: {0}", gpumat.Cols);
            Cuda.cuda.pyrUp(gpumat, gpumat_2);
            Cuda.cuda.pyrDown(gpumat_2, gpumat_2);
            //Cv2.Resize

            Mat des = new Mat();
            gpumat_2.Download(des);
            Cv2.ImWrite("test.png",des);

            //output.WriteLine("sdfsdf {0}", Cv2.GetCudaEnabledDeviceCount());
            Cv2.PrintCudaDeviceInfo(0);
            
        }

        [Fact]
        public void SimpleGPUStreamTest() {
            Cuda.GpuMat gpumat = new Cuda.GpuMat();
            Cuda.GpuMat gpumat_2 = new Cuda.GpuMat();
            Mat src = Image("lenna.png", ImreadModes.Grayscale);

            Cuda.Stream stream = new Cuda.Stream();

            gpumat.Upload(src, stream);

            output.WriteLine("out: {0}", gpumat.Cols);
            Cuda.cuda.Resize(gpumat, gpumat, new Size(120, 120), 0, 0, InterpolationFlags.Linear, stream);
            output.WriteLine("out: {0}", gpumat.Cols);
            Cuda.cuda.pyrUp(gpumat, gpumat_2, stream);
            Cuda.cuda.pyrDown(gpumat_2, gpumat_2, stream);
            //Cv2.Resize

            Mat des = new Mat();

            gpumat_2.Download(des, stream);
            stream.WaitForCompletion();

            Cv2.ImWrite("test.png", des);

            //output.WriteLine("sdfsdf {0}", Cv2.GetCudaEnabledDeviceCount());
            Cv2.PrintCudaDeviceInfo(0);

        }

        [Fact]
        public void SimplecannyTest() {
            Cuda.GpuMat gpumat = new Cuda.GpuMat();
            Cuda.GpuMat gpumat_2 = new Cuda.GpuMat();
            Mat src = Image("lenna.png", ImreadModes.Grayscale);
            gpumat.Upload(src);

            Cuda.CannyEdgeDetector canny = Cuda.CannyEdgeDetector.Create(100, 50);
            //bool aaaa = canny.Empty;
            //output.WriteLine(canny.ToString());
            output.WriteLine("out: {0}", canny.CvPtr);
            canny.detect(gpumat, gpumat_2);

            //canny.Dispose();

            Mat des = new Mat();
            gpumat_2.Download(des);
            Cv2.ImWrite("test.png", des);

        }

        [Fact]
        public void Simple2GPUTest() {
            Cuda.GpuMat gpumat = new Cuda.GpuMat();
            Cuda.GpuMat gpumat_des = new Cuda.GpuMat();
            Mat src = Image("lenna.png", ImreadModes.Grayscale);
            gpumat.Upload(src);

            Cuda.CannyEdgeDetector canny = Cuda.CannyEdgeDetector.Create(100, 50);

            canny.detect(gpumat, gpumat_des);

            Mat des = new Mat();
            gpumat_des.Download(des);
            Cv2.ImWrite("test.png", des);
        }

    }
}
