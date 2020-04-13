/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace TFLite
{

    /// <summary>
    /// Driver class to drive model inference with TensorFlow Lite. Interpreter
    /// encapsulates a pre-trained model file in whihc the operations are performed
    /// @class	Interpreter
    /// </summary>
    public class Interpreter : IDisposable
    {
        // Handle to hold the model instance
        private IntPtr m_modelHandle;
        // Handle to hold the interpreter instance
        private IntPtr m_interpreterHandle;

        /// <summary>
        /// Interpreter Constructor. Inititalizes an interpreter.     
        /// </summary>
        ///<param name="modelPath">a File of a pre-trained TF Lite model. </param>        
        public Interpreter(string modelPath)
        {
            //Constructor to initialize the interpreter with a model file
            m_modelHandle = Interop.TFLite.TFLiteFlatBufferModelBuildFromFile(modelPath);
            if(m_modelHandle == IntPtr.Zero)
            {
                //TODO: routine for handling null pointer.
            }
            m_interpreterHandle = Interop.TFLite.TFLiteBuilderInterpreterBuilder(ref m_modelHandle);
            if (m_interpreterHandle == IntPtr.Zero)
            {
                //TODO: routine for handling null pointer.
            }
        }

        /// <summary>
        /// Set the number of threads available to the interpreter.
        /// </summary>
        /// <param name="numThreads">Number of threads.</param>
        public void SetNumThreads(int numThreads)
        {
            Interop.TFLite.TFLiteInterpreterSetNumThreads(numThreads);
            return;
        }

        /// <summary>
        /// Runs model inference if the model takes only one input, and provides only
        /// one output.
        /// </summary>
        /// <param name="input">input an array or multidimensional array.</param>
        /// <param name="output">outputs a multidimensional array of output data.</param>
        public void Run(Array input, ref Array output)
        {
            Array[] inputs = { input };
            Dictionary<int, Array> outputs = new Dictionary<int, Array>();

            RunForMultipleInputsOutputs(inputs, ref outputs);
            output = outputs[0];

            return;
        }

        /// <summary>
        /// Runs model inference if the model takes multiple inputs, or returns multiple
        /// outputs.
        /// </summary>
        /// <param name="inputs">input an array of input data.</param>
        /// <param name="outputs">outputs a map mapping output indices to multidimensional
        /// arrays of output data.</param>
        public void RunForMultipleInputsOutputs(Array[] inputs, ref Dictionary<int, Array> outputs)
        {
            if(m_interpreterHandle == IntPtr.Zero)
            {
                //TODO:: exception handling
            }

            if (inputs == null || inputs.Length == 0)
            {
                //TODO::throw new IllegalArgumentException("Input error: Inputs should not be null or empty.");
            }

            DataType[] dataTypes = new DataType[inputs.Length];//To be used in multi-dimensional case

            for (int i = 0; i < inputs.Length; ++i)
            {
                dataTypes[i] = DataTypeOf(inputs[i]);
            }

            //TODO:: Support for multi dimesional array to be added.
            IntPtr pnt = Marshal.AllocHGlobal(inputs[0].Length);

            switch (dataTypes[0])
            {
                case DataType.INT32:
                    Marshal.Copy((int[])inputs[0], 0, pnt, inputs[0].Length);
                    break;
                case DataType.FLOAT32:
                    Marshal.Copy((float[])inputs[0], 0, pnt, inputs[0].Length);
                    break;
                case DataType.UINT8:
                    Marshal.Copy((byte[])inputs[0], 0, pnt, inputs[0].Length);
                    break;
                case DataType.INT64:
                    Marshal.Copy((long[])inputs[0], 0, pnt, inputs[0].Length);
                    break;
                default:
                    Marshal.Copy((byte[])inputs[0], 0, pnt, inputs[0].Length);
                    break;
            }

            //Currently this handles only single input with single dimension.
            IntPtr outputsHandles = Interop.TFLite.TFLiteInterpreterRun(ref m_interpreterHandle, pnt, inputs[0].Length, (int)dataTypes[0]);

            if (outputsHandles == null)
            {
                //throw new IllegalStateException("Internal error: Interpreter has no outputs.");
            }

            switch (dataTypes[0])
            {
                case DataType.INT32:
                    int[] managedArrayInt = new int[inputs[0].Length];
                    Marshal.Copy(outputsHandles, managedArrayInt, 0, inputs[0].Length);
                    outputs.Add(0, managedArrayInt);
                    break;
                case DataType.FLOAT32:
                    float[] managedArrayFloat = new float[inputs[0].Length];
                    Marshal.Copy(outputsHandles, managedArrayFloat, 0, inputs[0].Length);
                    outputs.Add(0, managedArrayFloat);
                    break;
                case DataType.UINT8:
                    byte[] managedArrayByte = new byte[inputs[0].Length];
                    Marshal.Copy(outputsHandles, managedArrayByte, 0, inputs[0].Length);
                    outputs.Add(0, managedArrayByte);
                    break;
                case DataType.INT64:
                    long[] managedArrayLong = new long[inputs[0].Length];
                    Marshal.Copy(outputsHandles, managedArrayLong, 0, inputs[0].Length);
                    outputs.Add(0, managedArrayLong);
                    break;
                default:
                    byte[] managedArrayDefault = new byte[inputs[0].Length];
                    Marshal.Copy(outputsHandles, managedArrayDefault, 0, inputs[0].Length);
                    outputs.Add(0, managedArrayDefault);
                    break;
            }
            return;
        }

        static DataType DataTypeOf(Array a)
        {
            if (a.GetValue(0).GetType()==typeof(int))
            {
                return DataType.INT32;
            }
            else if (a.GetValue(0).GetType() == typeof(float))
            {
                return DataType.FLOAT32;
            }
            else if (a.GetValue(0).GetType() == typeof(byte))
            {
                return DataType.UINT8;
            }
            else if(a.GetValue(0).GetType() == typeof(long))
            {
                return DataType.INT64;
            }
            else
            {
                return DataType.UINT8;
                //TODO: throw exception
            }

        }

        /// <summary>
        /// Resizes idx-th input of the native model to the given dims.
        /// </summary>
        /// <param name="idx">index of the input.</param>
        /// <param name="dims">Dimensions to which input needs to be resized.</param>
        public void ResizeInput(int idx, int[] dims)
        {
            return;
        }

        /// <summary>
        /// Gets index of an input given the tensor name of the input.
        /// </summary>
        /// <param name="tensorName">Name of the tensor.</param>
        public int GetInputIndex(string tensorName)
        {
            return 0;
        }

        /// <summary>
        /// Gets index of output given the tensor name of the input.
        /// </summary>
        /// <param name="tensorName">Name of the tensor.</param>
        public int GetOutputIndex(string tensorName)
        {
            return 0;
        }

        /// <summary>
        /// Turns on/off Android NNAPI for hardware acceleration when it is available.
        /// </summary>
        /// <param name="useNNAPI">set the boolean value to turn on/off nnapi.</param>
        public void SetUseNNAPI(bool useNNAPI)
        {
            return;
        }

        /// <summary>
        /// Release resources associated with the Interpreter.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
        }

        protected virtual void Dispose(bool bDisposing)
        {
            if (m_interpreterHandle != IntPtr.Zero)
            {
                // Call the function to dispose this class
                m_interpreterHandle = IntPtr.Zero;
            }

            if (bDisposing)
            {
                // No need to call the finalizer since we've now cleaned
                // up the unmanaged memory
                GC.SuppressFinalize(this);
            }
        }

        // This finalizer is called when Garbage collection occurs, but only if
        // the IDisposable.Dispose method wasn't already called.
        ~Interpreter()
        {
            Dispose(false);
        }
    }
}
