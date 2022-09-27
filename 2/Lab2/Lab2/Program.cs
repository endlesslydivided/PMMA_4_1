using Emgu.CV;
using Emgu.CV.CvEnum;
using System;
using System.Collections.Generic;
using System.Drawing;
using Emgu.CV.Structure;

namespace Lab2
{
    internal class Program
    {
        public enum Effect
        {
            ContrastHigh,
            Blur,
            GausianBlur,
            MedianBlur,
            Erode,
            Dilate
        }
        static void Main(string[] args)
        {
            Mat imageNormal = new Mat("image.jpg");
            Mat items = new Mat("items.jpg");
            Mat itemsAdaptive = new Mat();
            CvInvoke.CvtColor(items, items, ColorConversion.Bgr2Gray);
            CvInvoke.Threshold(items, items, 185, 255, ThresholdType.Binary);
            CvInvoke.AdaptiveThreshold(items, itemsAdaptive,255,AdaptiveThresholdType.GaussianC,ThresholdType.Binary,3,0);

            //1
            CvInvoke.Imshow("Normal", imageNormal);
            CvInvoke.Imshow("ContrastHigh", MakeEffect(imageNormal,Effect.ContrastHigh));
            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();

            //2
            CvInvoke.Imshow("Normal", imageNormal);
            CvInvoke.Imshow("GausianBlur", MakeEffect(imageNormal, Effect.GausianBlur));
            CvInvoke.Imshow("Blur", MakeEffect(imageNormal, Effect.Blur));
            CvInvoke.Imshow("MedianBlur", MakeEffect(imageNormal, Effect.MedianBlur));
            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();

            //3

            Mat dilate = MakeEffect(items, Effect.Dilate);
            Mat erode = MakeEffect(items, Effect.Erode);

            CvInvoke.Imshow("Normal", items);
            CvInvoke.Imshow("Erode", erode); ;
            CvInvoke.Imshow("Dilate", dilate);
            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();

            //4

            CvInvoke.Imshow("Normal - erode", items - erode);
            CvInvoke.Imshow("Normal - dilate", dilate - items);
            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();

            //5

            Mat erodeAdaptive = MakeEffect(itemsAdaptive, Effect.Erode);
            Mat dilateAdaptive = MakeEffect(itemsAdaptive, Effect.Dilate);

            CvInvoke.Imshow("Normal", itemsAdaptive);
            CvInvoke.Imshow("Erode", erodeAdaptive);
            CvInvoke.Imshow("Dilate", dilateAdaptive);
            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();

            CvInvoke.Imshow("Normal adaptive - erode", itemsAdaptive - erodeAdaptive);
            CvInvoke.Imshow("Normal adaptive - dilate", dilateAdaptive - itemsAdaptive);

            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();

        }

        static Mat MakeEffect(Mat image, Effect effect)
        {
            Mat result = new Mat(image.Size, DepthType.Cv64F,3);

            switch (effect)
            {
                case Effect.ContrastHigh:
                    {                      
                        Matrix<float> kernel = new Matrix<float>( new float[,] { { -1, -1, -1 }, { -1, 8.5f, -1 }, { -1, -1, -1 } }) ;

                        CvInvoke.Filter2D(image, result, kernel, new Point(-1,-1));

                        break;
                    }
                case Effect.GausianBlur:
                    {
                        CvInvoke.GaussianBlur(image, result, new Size(5, 5),1);
                        break;
                    }
                case Effect.MedianBlur:
                    {
                        CvInvoke.MedianBlur(image, result,9);
                        break;
                    }
                case Effect.Blur:
                    {
                        CvInvoke.Blur(image, result,new Size(5,5), new Point(-1, -1));
                        break;
                    }
                case Effect.Erode:
                    {
                        Matrix<float> kernel = new Matrix<float>(new float[,] 
                        { 
                            { 1,  1, 1    }, 
                            {  1,   5,  1    }, 
                            {  1,  1,  1    } 
                        });
                        CvInvoke.Erode(image, result, kernel, new Point(-1, -1), 5, BorderType.Default, new MCvScalar());
                        break;
                    }
                case Effect.Dilate:
                    {
                        Matrix<float> kernel = new Matrix<float>(new float[,]
                        {
                            { 1,  1, 1    },
                            {  1,   7,  1    },
                            {  1,  1,  1    }
                        });
                        CvInvoke.Dilate(image, result, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar());
                        break;
                    }

            }

            return result;
        }
    }
}
