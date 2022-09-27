using Emgu.CV;
using System;
using System.Drawing;
using static Emgu.CV.CvEnum.DepthType;
using static Emgu.CV.CvEnum.ColorConversion;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Collections.Generic;

namespace Lab3
{
    internal class Program
    {
        static Mat Angle(Mat image)
        {
            Mat work = image.Clone();

            Mat grey = new Mat();
            Mat binary = new Mat();
            Mat canny_output = new Mat();
            Mat mat = new Mat();


            CvInvoke.CvtColor(image, grey, ColorConversion.Bgr2Gray);
            CvInvoke.MedianBlur(grey, grey, 5);

            CvInvoke.Threshold(grey, binary, 168, 250, ThresholdType.Binary);


            CvInvoke.Canny(binary, canny_output, 65, 40);
            VectorOfMat contours = new VectorOfMat();
            Mat hierarchy = new Mat();

            CvInvoke.FindContours(canny_output, contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxTc89L1);
            Mat drawing = Mat.Zeros(canny_output.Rows, canny_output.Cols, DepthType.Cv64F, 3);

            Dictionary<int, int> areas = new Dictionary<int, int>();
            for (int i = 0; i < int.Parse(contours.Size.ToString()); i++)
            {
                RotatedRect rect = CvInvoke.MinAreaRect(contours[i]);             
                int area = (int)rect.Size.Width * (int)rect.Size.Height;

                areas.Add(i, area);

            }
            int iter = 0;
            int maxArea = 0;
            foreach (KeyValuePair<int,int> item in areas)
            {
                if(item.Value > maxArea)
                {
                    iter = item.Key;
                }
            }
            RotatedRect rectToUse = CvInvoke.MinAreaRect(contours[iter]);
            PointF[] box = CvInvoke.BoxPoints(rectToUse);

            int[] edge1 = { (int)box[0].X - (int)box[1].X, (int)box[0].Y - (int)box[1].Y };

            VectorOfInt reference = new VectorOfInt(new int[] { 1, 0 });
            VectorOfInt edgeVector = new VectorOfInt(edge1);

            double angle = 180 / Math.PI * Math.Acos((1 * Math.Abs(edge1[0]) + 0 * Math.Abs(edge1[1])) / (CvInvoke.Norm(reference) * CvInvoke.Norm(edgeVector)));
            angle = Math.Round(angle, 3);

            CvInvoke.DrawContours(work, contours, iter, new MCvScalar(0, 0, 255), 2, LineType.EightConnected);
            CvInvoke.Circle(work, new Point((int)rectToUse.Center.X, (int)rectToUse.Center.Y), 5, new MCvScalar(255, 0, 0));
            CvInvoke.PutText(work, angle.ToString(), new Point((int)rectToUse.Center.X + 20, (int)rectToUse.Center.Y + 20), FontFace.HersheySimplex, 1, new MCvScalar(0, 0, 250));

            return work;
        }


        static void Main(string[] args)
        {
            VideoCapture videoCapture = new VideoCapture();

            if (!videoCapture.IsOpened)
            {
                Console.WriteLine("Error!");
                return;
            }

            Mat frame = new Mat();
            Mat sobel = new Mat();
            Mat laplas = new Mat();
            Mat canny = new Mat();

            while (true)
            {
                videoCapture.Read(frame);

                CvInvoke.GaussianBlur(frame, frame, new Size(5, 5), 0);

                CvInvoke.Sobel(frame, sobel, Default, 1, 0);
                CvInvoke.Laplacian(frame, laplas, Default,5,1);
                CvInvoke.CvtColor(frame, canny, Bgr2Gray);
                CvInvoke.Canny(canny, canny, 40, 40);

                CvInvoke.Imshow("Frame", frame);
                //CvInvoke.Imshow("Sobel", sobel);
                //CvInvoke.Imshow("Laplas", laplas);
                //CvInvoke.Imshow("Canny", canny);
                CvInvoke.Imshow("Angle", Angle(frame));


                int keyboard = CvInvoke.WaitKey(30);

                if (keyboard == 'q' || keyboard == 27)
                {
                    break;
                }
            }
            return;

        }
    }
}
