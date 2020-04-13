using System;

namespace TFLiteSharpTest
{
    class Program
    {
        static void Main(string[] args)
        {
            //Constructing a new interpreter instance from the modelfile
            TFLite.Interpreter interpreter = new TFLite.Interpreter("modelpath/modelfile.tflite");
            Console.WriteLine("Interpreter Built Successfully");

            //Setting the number of threads of the interpreter
            interpreter.SetNumThreads(1);

            //Declaring input and output variables;
            Array input = new int[5] { 1, 2, 3, 4, 5 };
            Array output = new int[5];

            //Call to invoke the interpreter and run the inference to populate output
            interpreter.Run(input, out output);
            Console.WriteLine("Output generated Successfully");

            //get input, output indices
            Console.WriteLine("Input index for tensorname: " + interpreter.GetInputIndex("tensorname"));
            Console.WriteLine("Output index for tensorname: " + interpreter.GetOutputIndex("tensorname"));

            //Resizing the dimensions
            int[] dims = new int[3] { 1, 2, 3 };
            interpreter.ResizeInput(1, dims);

            //Disposing the interpreter to free resources at the end
            interpreter.Dispose();

            Console.WriteLine("Run Complete");
        }
    }
}
