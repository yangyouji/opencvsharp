﻿using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

[assembly: CollectionBehavior(/*MaxParallelThreads = 2, */DisableTestParallelization = true)]

namespace OpenCvSharp.Tests
{
    public abstract class TestBase
    {
        private readonly HttpClient httpClient;
        protected readonly ITestOutputHelper output;

        public TestBase()
        {
            ServicePointManager.ServerCertificateValidationCallback = (sender, certificate, chain, sslPolicyErrors) => true;
#if DOTNET_FRAMEWORK
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls11 | SecurityProtocolType.Tls12;
#endif

            httpClient = new HttpClient
            {
                Timeout = TimeSpan.FromMinutes(5)
            };
        }

        public TestBase(ITestOutputHelper output) {
            ServicePointManager.ServerCertificateValidationCallback = (sender, certificate, chain, sslPolicyErrors) => true;
#if NET461
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls11 | SecurityProtocolType.Tls12;
#endif
            this.output = output;
            httpClient = new HttpClient {
                Timeout = TimeSpan.FromMinutes(5)
            };
        }

        protected static Mat Image(string fileName, ImreadModes modes = ImreadModes.Color)
        {
            return new Mat(Path.Combine("_data", "image", fileName), modes);
        }

        protected static void ImageEquals(Mat img1, Mat img2)
        {
            if (img1 == null && img2 == null)
                return;
            Assert.NotNull(img1);
            Assert.NotNull(img2);
            Assert.Equal(img1.Type(), img2.Type());

            using (var comparison = new Mat())
            {
                Cv2.Compare(img1, img2, comparison, CmpTypes.NE);
                if (img1.Channels() == 1)
                {
                    Assert.Equal(0, Cv2.CountNonZero(comparison));
                }
                else
                {
                    var channels = Cv2.Split(comparison);
                    try
                    {
                        foreach (var channel in channels)
                        {
                            Assert.Equal(0, Cv2.CountNonZero(channel));
                        }
                    }
                    finally
                    {
                        foreach (var channel in channels)
                        {
                            channel.Dispose();
                        }
                    }
                }
            }
        }

        protected static void ImageEquals(Mat img1, Mat img2, double abs_error) {
            if (img1 == null && img2 == null)
                return;
            Assert.NotNull(img1);
            Assert.NotNull(img2);
            Assert.Equal(img1.Type(), img2.Type());
            double abs_sum = abs_error * img1.Width * img1.Height;

            using (var comparison = new Mat()) {
                Cv2.Absdiff(img1, img2, comparison);
                if (img1.Channels() == 1) {
                    Assert.False(Cv2.Sum(comparison).Val0 > abs_sum);
                } else {
                    var channels = Cv2.Split(comparison);
                    try {
                        foreach (var channel in channels) {
                            Assert.False(Cv2.Sum(channel).Val0 > abs_sum);
                        }
                    } finally {
                        foreach (var channel in channels) {
                            channel.Dispose();
                        }
                    }
                }
            }
        }

        protected static void ShowImagesWhenDebugMode(params Mat[] mats)
        {
            if (Debugger.IsAttached)
            {
                Window.ShowImages(mats);
            }
        }

        protected static void Pause(string message = "Press any key to exit")
        {
            if (Debugger.IsAttached)
            {
                Console.WriteLine();
                Console.WriteLine(message);
                Console.Read();
            }
        }

        protected static byte[] DownloadBytes(string url)
        {
            using (var client = new MyWebClient())
                return client.DownloadData(url);
            //var response = (await httpClient.GetAsync(url)).EnsureSuccessStatusCode();
            //return await response.Content.ReadAsByteArrayAsync();
        }

        protected static string DownloadString(string url)
        {
            using (var client = new MyWebClient())
                return client.DownloadString(url);
            //var response = (await httpClient.GetAsync(url)).EnsureSuccessStatusCode();
            //return await response.Content.ReadAsStringAsync();
        }

        private class MyWebClient : WebClient
        {
            protected override WebRequest GetWebRequest(Uri uri)
            {
                WebRequest w = base.GetWebRequest(uri);
                w.Timeout = 5 * 60 * 1000; // ms
                return w;
            }
        }
    }
}
