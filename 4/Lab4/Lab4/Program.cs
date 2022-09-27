using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Lab2;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Numerics;
using static Lab2.Methods;

namespace Lab4
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Mat image = new Mat("items.jpg");
            Mat lines = new Mat("lines2.jpg");
            Mat circles = new Mat("circles.jpg");
            Mat list = new Mat("list.jpg");

            //1
            Contours(image);

            //2
            Lines(lines);

            //3
            Circles(circles);

            //4
            Angle(list);


        }

        static void Circles(Mat image)
        {
            Mat grey = image.Clone();
            Mat circles = image.Clone();

            CvInvoke.CvtColor(image, grey, ColorConversion.Bgr2Gray);
            Mat drawing = Mat.Zeros(image.Rows, image.Cols, DepthType.Cv8U,1);
            CvInvoke.HoughCircles(grey, circles,HoughModes.Gradient,1,10,250,150);

            for (int i = 0; i < circles.Cols; ++i)
            {
                float x = (float)circles.GetData().GetValue(0,i,0);
                float y = (float)circles.GetData().GetValue(0,i,1);
                float radius = (float)circles.GetData().GetValue(0,i, 2);
                Point center = new Point((int)x, (int)y);
                CvInvoke.Circle(image, center, (int)radius, new MCvScalar(0,0,250),2);
            }

            CvInvoke.Imshow("Normal", image);

            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();
        }

        static void Angle(Mat image)
        {
            Mat work = image.Clone();

            Mat grey = new Mat();
            Mat binary = new Mat();
            Mat canny_output = new Mat();


            CvInvoke.CvtColor(image, grey, ColorConversion.Bgr2Gray);
            CvInvoke.MedianBlur(grey, grey, 5);
            CvInvoke.Threshold(grey, binary, 168, 220, ThresholdType.Binary);
            CvInvoke.Canny(binary, canny_output, 65, 40);

            VectorOfMat contours = new VectorOfMat();
            Mat hierarchy = new Mat();

            CvInvoke.FindContours(canny_output, contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxTc89L1);
            Mat drawing = Mat.Zeros(canny_output.Rows, canny_output.Cols, DepthType.Cv64F, 3);
            for (int i = 0; i < int.Parse(contours.Size.ToString()); i++)
            {
                RotatedRect rect = CvInvoke.MinAreaRect(contours[i]);
                PointF[] box = CvInvoke.BoxPoints(rect);
                int[] edge1 = { (int)box[0].X - (int)box[1].X, (int)box[0].Y - (int)box[1].Y };
                int[] edge2 = { (int)box[2].X - (int)box[3].X, (int)box[2].Y - (int)box[3].Y };

                VectorOfInt reference = new VectorOfInt(new int[] { 1,0});
                VectorOfInt edgeVector = new VectorOfInt(edge1);

                int area = (int)rect.Size.Width * (int)rect.Size.Height;
                
                double angle = 180 / Math.PI * Math.Acos((1 * Math.Abs(edge1[0]) + 0 * Math.Abs(edge1[1])) / (CvInvoke.Norm(reference) * CvInvoke.Norm(edgeVector)));
                angle = Math.Round(angle, 3);
                if(area >= 145801)
                {
                    CvInvoke.DrawContours(work, contours,i, new MCvScalar(0, 0, 255),2,LineType.EightConnected);
                    CvInvoke.Circle(work, new Point((int)rect.Center.X, (int)rect.Center.Y), 5, new MCvScalar(255, 0, 0));
                    CvInvoke.PutText(work, angle.ToString(), new Point((int)rect.Center.X + 20, (int)rect.Center.Y + 20), FontFace.HersheySimplex, 1, new MCvScalar(0, 255, 255));
                }    
            }

            CvInvoke.Imshow("Result", work);

            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();
        }

        static void Contours(Mat image)
        {
            Mat canny_output = new Mat();
            Mat blur = new Mat();
            Mat grey = new Mat();
            Mat binary = new Mat();

            CvInvoke.MedianBlur(image, blur,5);
            CvInvoke.CvtColor(blur, grey, ColorConversion.Bgr2Gray);
            CvInvoke.Threshold(grey, binary,100, 250, ThresholdType.Binary);

            CvInvoke.Canny(binary, canny_output,55,20);
            Random random = new Random();

            VectorOfMat contours = new VectorOfMat();
            Mat hierarchy = new Mat();

            CvInvoke.FindContours(canny_output, contours, hierarchy, RetrType.External,ChainApproxMethod.ChainApproxSimple);
           
            Mat drawing = Mat.Zeros(canny_output.Rows,canny_output.Cols,DepthType.Cv64F,3);
            for (int i = 0; i < int.Parse(contours.Size.ToString()); i++)
            {
                MCvScalar color = new MCvScalar(255,0,0);
                CvInvoke.DrawContours(drawing, contours, (int)i, color,1,LineType.EightConnected, hierarchy);
            }

            CvInvoke.Imshow("Normal", image);
            CvInvoke.Imshow($"Drawing", drawing);
            CvInvoke.Imshow($"canny_output", canny_output);

            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();
        }

        static void Lines(Mat image)
        {
            Mat canny_output = new Mat();
            Mat RGB = new Mat();
            Mat blur = new Mat();
            Mat erodeDilate = new Mat();
            Mat binary = new Mat();

            Mat drawing = image.Clone();
            Mat drawingC = image.Clone();

            CvInvoke.CvtColor(image, blur, ColorConversion.Bgr2Gray);

            CvInvoke.MedianBlur(blur, blur,3);
            //CvInvoke.AdaptiveThreshold(blur, binary,45,AdaptiveThresholdType.GaussianC, ThresholdType.Binary,3,10);

            CvInvoke.Canny(blur, canny_output, 40, 50);

            //canny_output = Methods.MakeEffect(canny_output, Effect.Dilate);
            //canny_output = Methods.MakeEffect(canny_output, Effect.Erode);

            Mat RGBC = canny_output.Clone();
            Random random = new Random();

            Mat lines = new Mat();

            //2.1
            CvInvoke.HoughLines(canny_output, lines, 1, 3.14 / 180, 205);

            Dictionary<int,double> thetaOne = new Dictionary<int, double>();
            for (int i = 0; i < lines.Width * lines.Height; i++)
            {
                float rho = (float)lines.GetData().GetValue(i, 0, 0),theta = (float)lines.GetData().GetValue(i, 0, 1);
                Point pt1 = new Point(), pt2 = new Point();
                double a = Math.Cos(theta), b = Math.Sin(theta);
                double x0 = a * rho, y0 = b * rho;
                pt1.X = (int)Math.Round(x0 + 1000 * (-b));
                pt1.Y = (int)Math.Round(y0 + 1000 * (a));
                pt2.X = (int)Math.Round(x0 - 1000 * (-b));
                pt2.Y = (int)Math.Round(y0 - 1000 * (a));
                MCvScalar mCvScalar = new MCvScalar(0, 0, 255);
                int thickness = 1;

                foreach (KeyValuePair<int,double> item in thetaOne)
                {
                    if (Math.Round(item.Value, 1) == Math.Round(theta, 1))
                    {
                        int iter = item.Key;
                        mCvScalar = new MCvScalar((iter % 10 == 0) ? iter * 4 : iter * 5, iter * 10, (iter % 2 == 0) ? iter * 2 : iter * 3);
                        thickness = 1;
                    }
                }
                thetaOne.Add(i, theta);

                CvInvoke.Line(drawing, pt1, pt2, mCvScalar, thickness, LineType.EightConnected);
            }

            //2.1
            CvInvoke.HoughLinesP(canny_output, lines, 1, 3.14 / 180, 100);
            for (int i = 0; i < lines.Width * lines.Height; i++)
            {

                int x1 = (int)lines.GetData().GetValue(i, 0, 0);
                int y1 = (int)lines.GetData().GetValue(i, 0, 1);
                int x2 = (int)lines.GetData().GetValue(i, 0, 2);
                int y2 = (int)lines.GetData().GetValue(i, 0, 3);

                CvInvoke.Line(drawingC,new Point(x1, y1),new Point(x2, y2), new MCvScalar(0, 0, 255), 1, LineType.EightConnected);
            }


            CvInvoke.Imshow("Normal", image);
            CvInvoke.Imshow("Canny", canny_output);
            CvInvoke.Imshow($"Drawing", drawing);
            CvInvoke.Imshow($"DrawingC", drawingC);

            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();
        }
    }      
}

