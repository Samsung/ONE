/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <iostream>
#include "tflite_nativewrapper.h"
#include "tflite_log.h"
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

int getNumBytes(TFLiteNativeType dataType)
{
  switch (dataType)
  {
    case INT32:
      return 4;
    case FLOAT32:
      return 4;
    case UINT8:
      return 1;
    case INT64:
      return 8;
    default:
      return 1;
  }
}

/// <summary>
/// Set the number of threads available to the interpreter.
/// </summary>
/// <param name="interpreterHandle">Handle of the interpreter instance.</param>
/// <param name="numThreads">Number of threads.</param>
void tflite_interpreter_setNumThreads(long *interpreterHandle, int numThreads)
{
  assert(interpreterHandle != nullptr);
  tflite::Interpreter *interpreter = reinterpret_cast<tflite::Interpreter *>(*interpreterHandle);

  interpreter->SetNumThreads(numThreads);

  TFLITE_NATIVE_LOG(DEBUG, "Number of threads: %d", numThreads);
  return;
}

/// <summary>
/// Creates a Flat Buffer Model from the given .tflite model.
/// </summary>
/// <param name="modelPath">Path of the model.</param>
long long tflite_flatbuffermodel_BuildFromFile(char *modelPath)
{
  if (modelPath == nullptr)
  {
    TFLITE_NATIVE_LOG(ERROR, "Invalid parameter");
    return 0;
  }
  TFLITE_NATIVE_LOG(ERROR, "Model Path: %s", modelPath);

  if (access(modelPath, F_OK) == -1)
  {
    TFLITE_NATIVE_LOG(ERROR, "Failed to access model [%s]", strerror(errno));
    return 0;
  }

  auto model = tflite::FlatBufferModel::BuildFromFile(modelPath);

  TFLITE_NATIVE_LOG(DEBUG, "Successfully loaded model");
  return reinterpret_cast<long>(model.release());
}

/// <summary>
/// Creates an interpreter instance taking the flatbuffer model as input.
/// </summary>
/// <param name="modelHandle">Address of the flatbuffer model.</param>
long long tflite_builder_interpreterBuilder(long *modelHandle)
{
  assert(modelHandle != nullptr);
  tflite::FlatBufferModel *model = reinterpret_cast<tflite::FlatBufferModel *>(*modelHandle);

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;

  TfLiteStatus status = tflite::InterpreterBuilder(*model, resolver)(&interpreter);

  if (status != kTfLiteOk)
  {
    TFLITE_NATIVE_LOG(DEBUG, "Cannot create interpreter");
    return 0;
  }
  TFLITE_NATIVE_LOG(DEBUG, "CheckPoint interpreter");
  return reinterpret_cast<long>(interpreter.release());
}

/// <summary>
/// Runs the inference given the inputs.
/// </summary>
/// <param name="interpreterHandle">Address of the interpreter instance.</param>
/// <param name="values">Input values for the model.</param>
/// <param name="inpLength">Length of the input.</param>
/// <param name="dataType">Data type key of the input.</param>
void *tflite_interpreter_run(long *interpreterHandle, void *values, int inputLength, int dataType)
{
  assert(interpreterHandle != nullptr);
  tflite::Interpreter *interpreter = reinterpret_cast<tflite::Interpreter *>(*interpreterHandle);

  int inputTensorIndex = interpreter->inputs()[0];

  // TODO:: input tensor size will be passed as a parameter. It is hardcoded for now.
  interpreter->ResizeInputTensor(inputTensorIndex, {1, 224, 224, 3});

  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    TFLITE_NATIVE_LOG(ERROR, "Failed to allocate tensors!");
    return nullptr;
  }

  float *inputTensorPointer = interpreter->typed_tensor<float>(inputTensorIndex);

  int numBytes = getNumBytes((TFLiteNativeType)dataType);

  memcpy(inputTensorPointer, values, inputLength * numBytes);

  if (interpreter->Invoke() != kTfLiteOk)
  {
    TFLITE_NATIVE_LOG(ERROR, "Failed to invoke");
  }

  float *output = interpreter->typed_output_tensor<float>(0);
  return output;
}
