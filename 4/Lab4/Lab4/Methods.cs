using Emgu.CV;
using Emgu.CV.CvEnum;
using System;
using System.Collections.Generic;
using System.Drawing;
using Emgu.CV.Structure;

namespace Lab2
{
    internal class Methods
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


        public static Mat MakeEffect(Mat image, Effect effect)
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
                        CvInvoke.Erode(image, result, kernel, new Point(-1, -1), 54, BorderType.Default, new MCvScalar());
                        break;
                    }
                case Effect.Dilate:
                    {
                        Matrix<float> kernel = new Matrix<float>(new float[,]
                        {
                            { 1,  1, 1    },
                            {  1,   5,  1    },
                            {  1,  1,  1    }
                        });
                        CvInvoke.Dilate(image, result, kernel, new Point(-1, -1), 75, BorderType.Default, new MCvScalar());
                        break;
                    }

            }

            return result;
        }
    }
}
