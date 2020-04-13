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
using System.Runtime.InteropServices;

internal static partial class Interop
{
    internal static partial class TFLite
    {
        [DllImport(Libraries.TFLite, EntryPoint = "tflite_flatbuffermodel_BuildFromFile")]
        internal static extern IntPtr TFLiteFlatBufferModelBuildFromFile(string path);

        [DllImport(Libraries.TFLite, EntryPoint = "tflite_builder_interpreterBuilder")]
        internal static extern IntPtr TFLiteBuilderInterpreterBuilder(ref IntPtr modelHandle);

        [DllImport(Libraries.TFLite, EntryPoint = "tflite_interpreter_setNumThreads")]
        internal static extern void TFLiteInterpreterSetNumThreads(int numThreads);

        [DllImport(Libraries.TFLite, EntryPoint = "tflite_interpreter_run")]
        internal static extern IntPtr TFLiteInterpreterRun(ref IntPtr interpreterHandle, IntPtr values, int inpLen, int dataType);

    }
}
