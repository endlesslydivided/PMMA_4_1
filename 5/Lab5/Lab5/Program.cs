using System;
using System.Drawing;
using OpenCvSharp;
namespace Lab5
{
    internal class Program
    {
        public static Scalar red = new Scalar(0, 0, 255);
        public static int thresh = 104;
        public static int max_tresh = 255;
        public static Mat lines = new Mat("list.jpg");
        public static Mat lines2 = new Mat("list.jpg");

        static void Main(string[] args)
        {

            Mat afine = new Mat("afine.jpg");

            Cv2.ImShow("Normal", lines);

            //1
            Cv2.NamedWindow("Corners");
            TrackbarCallbackNative trackbarCallbackNative = new TrackbarCallbackNative(HarrisCorner);
            Cv2.CreateTrackbar("Threshold: ", "Corners",ref thresh,max_tresh, trackbarCallbackNative);

            HarrisCorner(0, new IntPtr());

            //2
            ShiTomacy(lines2.Clone());

            //3
            Afine(afine.Clone());

            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();

        }

        static void HarrisCorner(int intv, IntPtr voidv)
        {
            int blockSize = 3;
            int apertureSize = 5;
            double k = 0.04;
            // определение углов
            Mat dst = Mat.Zeros(lines.Rows, lines.Cols, MatType.CV_32FC3);

            Mat srcGray = new Mat();
            Cv2.CvtColor(lines, srcGray, ColorConversionCodes.BGR2GRAY);


            Cv2.CornerHarris(srcGray, dst, blockSize, apertureSize, k);
            // нормализация выходного вектора углов
            Mat dstNorm = new Mat();
            Mat dstScaled = new Mat();

            Cv2.Normalize(dst, dstNorm, 0, 255, NormTypes.MinMax, MatType.CV_32FC1, new Mat());
            Cv2.ConvertScaleAbs(dstNorm, dstScaled);

            // рисование кругов вокруг углов
            // параметр thresh определяет порог отсечения (0-255)
            // чем он меньше, тем больше точек будет отрисовано
            Mat result = lines.Clone();
            for (int i = 0; i < dstNorm.Rows; i++)
            {
                for (int j = 0; j < dstNorm.Cols; j++)
                {
                    if (dstNorm.At<float>(i, j) > thresh)
                    {
                        Cv2.Circle(result, new OpenCvSharp.Point(j, i), 7, red, 1);
                    }
                }
            }
            Cv2.ImShow("Corners", result);

        }

        static void ShiTomacy(Mat image)
        {
            int blockSize = 3;
            double k = 0.04;


            (Mat draw, Mat contour) = Rect(image);


            Point2f[] corners = Cv2.GoodFeaturesToTrack(draw, 5, 0.00001, 50, new Mat(), blockSize, false, k);

            for (int i = 0; i < corners.Length; i++)
            {
                Cv2.Circle(image, (int)corners[i].X, (int)corners[i].Y, 10, red, 1);
            }
            Cv2.ImShow("ShiTomacyCorner", image);
        }

        static void Afine(Mat image)
        {
            int blockSize = 2;
            double k = 0.04;


            Mat srcGray = new Mat();
            Mat perspective = new Mat(image.Rows,image.Cols, MatType.CV_32F);
            Cv2.CvtColor(image, srcGray, ColorConversionCodes.BGR2GRAY);

            (Mat draw, Mat contour) = Rect(image);

            Point2f[] corners = Cv2.GoodFeaturesToTrack(draw, 4, 0.01, 250, new Mat(), blockSize, false, k);

            for (int i = 0; i < corners.Length; i++)
            {
                Cv2.Circle(image, (int)corners[i].X, (int)corners[i].Y, 10, red, 1);
            }

            Point2f[] src_vertices = new Point2f[4];
            src_vertices[0] = corners[0];
            src_vertices[1] = corners[1];
            src_vertices[2] = corners[2];
            src_vertices[3] = corners[3];

            Point2f[] dst_vertices = new Point2f[4];
            dst_vertices[3] = new OpenCvSharp.Point(1,1 );
            dst_vertices[1] = new OpenCvSharp.Point(image.Width - 1, 1);
            dst_vertices[2] = new OpenCvSharp.Point(1, image.Height - 1);
            dst_vertices[0] = new OpenCvSharp.Point(image.Width - 1, image.Height - 1);

            Mat warpAffineMatrix = Cv2.GetPerspectiveTransform(src_vertices, dst_vertices);


            Cv2.WarpPerspective(image, perspective, warpAffineMatrix, perspective.Size());
            Cv2.ImShow("Afine", image);
            Cv2.ImShow("Perspective", perspective);


        }

        static (Mat, Mat) Rect(Mat image)
        {
            Mat work = image.Clone();

            Mat grey = new Mat();
            Mat binary = new Mat();
            Mat canny_output = new Mat();


            Cv2.CvtColor(image, grey, ColorConversionCodes.BGR2GRAY);
            Cv2.MedianBlur(grey, grey, 5);
            Cv2.Threshold(grey, binary, 168, 220,ThresholdTypes.Binary);
            Cv2.Canny(binary, canny_output, 65, 40);

            Mat hierarchy = new Mat();

            Cv2.FindContours(canny_output, out Mat[] contours, hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
            Scalar color = new Scalar(255, 0, 0);
            Point2f[] box = null;
            RotatedRect rect = default;
            Mat contour = image.Clone();
            Mat drawing = Mat.Zeros(canny_output.Rows, canny_output.Cols, MatType.CV_32FC1);
            for (int i = 0; i < contours.Length; i++)
            {
                rect = Cv2.MinAreaRect(contours[i]);
                box = Cv2.BoxPoints(rect);
                if (rect.Size.Width * rect.Size.Height > 1000)
                {
                    contour = contours[i];
                    Cv2.DrawContours(drawing, contours, (int)i, color, 2, LineTypes.Link8);
                }
                
            }

            Cv2.ImShow($"Drawing", drawing);


            return (drawing, contour);
            
        }

    }
}
