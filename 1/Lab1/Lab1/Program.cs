using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using static Emgu.CV.CvEnum.ThresholdType;
using static Emgu.CV.CvEnum.DepthType;
using static Emgu.CV.CvEnum.NormType;
using static Emgu.CV.CvEnum.LineType;
using static Emgu.CV.CvEnum.ColorConversion;
using Emgu.CV.Util;
using System.Drawing;

namespace Lab1
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //2
            Mat imageNormal = new Mat("image.jpg");
            Mat imageGrey = new Mat();
            Mat imageBinary = new Mat();

            CvInvoke.CvtColor(imageNormal,imageGrey,Bgr2Gray);
            CvInvoke.Threshold(imageGrey, imageBinary, 140, 255,Binary);

            CvInvoke.Imshow("Normal picture", imageNormal);
            CvInvoke.Imshow("Grey picture", imageGrey);
            CvInvoke.Imshow("Binary picture", imageBinary);
            
            CvInvoke.WaitKey(0);

            CvInvoke.DestroyAllWindows();

            //3
            CvInvoke.Imwrite("grey.jpg",imageGrey);
            CvInvoke.Imwrite("binary.jpg", imageBinary);

            //4
            Mat imageBright = new Mat("imageBright.jpg");
            Mat imageGrayBright = new Mat();
            Mat imageGrayNormal = new Mat();
            
            CvInvoke.CvtColor(imageBright, imageGrayBright, Bgr2Gray);

            CvInvoke.EqualizeHist(imageGrayBright, imageGrayNormal);

        
            CvInvoke.Imshow("Source bright image", imageGrayBright);
            CvInvoke.Imshow("Histogram bright image", ShowHistogram(imageGrayBright));

            CvInvoke.Imshow("Source equelized bright image", imageGrayNormal);
            CvInvoke.Imshow("Histogram equelized bright image", ShowHistogram(imageGrayNormal));


            CvInvoke.WaitKey(0);

        }

        static Mat ShowHistogram(Mat image)
        {
            VectorOfMat bgrPlanes = new VectorOfMat();
            CvInvoke.Split(image, bgrPlanes);

            int histSize = 256;
            float[] range = { 0, 256 }; 

            bool accumulate = true;

            Mat bHist = new Mat();

            CvInvoke.CalcHist(bgrPlanes, new int[] { 0 }, new Mat(), bHist, new int[] {histSize}, range, accumulate);


            int histW = 512, histH = 400;
            int binW = (int)Math.Round((double)histW / histSize);
            Mat histImage = new Mat(histH, histW, Cv64F, 3);

            CvInvoke.Normalize(bHist, bHist, 0, histImage.Rows, MinMax);

            
            for (int i = 1; i < histSize; i++)
            {
                CvInvoke.Line(histImage, 
                    new Point(binW * (i - 1), (int)(histH - Math.Round((float)bHist.GetData().GetValue(i - 1, 0)))),
                    new Point(binW * (i), (int)(histH - Math.Round((float)bHist.GetData().GetValue(i,0)))),
                    new MCvScalar(255, 0, 0));
            }
            return histImage;
            
        }
    }
}
