using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Xamarin.Forms;

namespace TFLiteTestApp
{
    public class App : Application
    {
        public App()
        {
            TFLite.Interpreter interpreter = null;
            try
            {
                interpreter = new TFLite.Interpreter(Tizen.Applications.Application.Current.DirectoryInfo.Resource + "mobilenet_v1_1.0_224.tflite");
            }
            catch(Exception e)
            {
                Tizen.Log.Debug("tflite", "Error: " + e);
            }

            Tizen.Log.Debug("tflite", "Interpreter Initialised");
            Array Output = new byte[1000];

            Array input = new byte[150582];
            input = File.ReadAllBytes(Tizen.Applications.Application.Current.DirectoryInfo.Resource + "mouse_224.bmp");

            interpreter.Run(input, ref Output);
            //val variable to check if the Output array is being populated or not.
            byte val = ((byte[])Output)[0];
            // The root page of your application
            MainPage = new ContentPage
            {
                Content = new StackLayout
                {
                    VerticalOptions = LayoutOptions.Center,
                    Children = {
                        new Label {
                            HorizontalTextAlignment = TextAlignment.Center,
                            Text = "Welcome to Xamarin Forms!"
                        }
                    }
                }
            };
        }

        protected override void OnStart()
        {
            // Handle when your app starts
        }

        protected override void OnSleep()
        {
            // Handle when your app sleeps
        }

        protected override void OnResume()
        {
            // Handle when your app resumes
        }
    }
}
