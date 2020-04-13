/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tflite/ext/kernels/SquaredDifference.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#include <iostream>

namespace nnfw
{
namespace tflite
{
namespace custom
{
namespace SquaredDifference
{

void *InitSquaredDifference(TfLiteContext *, const char *, size_t) { return nullptr; }

void FreeSquaredDifference(TfLiteContext *, void *) {}

TfLiteStatus PrepareSquaredDifference(TfLiteContext *context, TfLiteNode *node)
{
  TF_LITE_ENSURE_EQ(context, ::tflite::NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, ::tflite::NumOutputs(node), 1);

  const TfLiteTensor *input1 = ::tflite::GetInput(context, node, 0);
  const TfLiteTensor *input2 = ::tflite::GetInput(context, node, 1);
  TfLiteTensor *output = ::tflite::GetOutput(context, node, 0);

  TF_LITE_ENSURE_EQ(context, input1->type, input2->type);
  TF_LITE_ENSURE_EQ(context, input1->type, output->type);

  return context->ResizeTensor(context, output, TfLiteIntArrayCopy(input1->dims));
}

TfLiteStatus EvalSquaredDifference(TfLiteContext *context, TfLiteNode *node)
{

  const TfLiteTensor *input1 = ::tflite::GetInput(context, node, 0);
  const TfLiteTensor *input2 = ::tflite::GetInput(context, node, 1);

  TfLiteTensor *output = ::tflite::GetOutput(context, node, 0);

  size_t elements = ::tflite::NumElements(input1);

  switch (input1->type)
  {
    case kTfLiteFloat32:
    {
      const float *in1 = input1->data.f;
      const float *in2 = input2->data.f;
      const float *in_end1 = in1 + elements;
      float *out = output->data.f;

      for (; in1 < in_end1; in1++, in2++, out++)
        *out = ((*in1 - *in2) * (*in1 - *in2));

      return kTfLiteOk;
    }
    case kTfLiteInt32:
    {
      const int *in1 = input1->data.i32;
      const int *in2 = input2->data.i32;
      const int *in_end1 = in1 + elements;
      int *out = output->data.i32;

      for (; in1 < in_end1; in1++, in2++, out++)
        *out = ((*in1 - *in2) * (*in1 - *in2));

      return kTfLiteOk;
    }
    case kTfLiteInt64:
    {
      const int64_t *in1 = input1->data.i64;
      const int64_t *in2 = input1->data.i64;
      const int64_t *in_end1 = in1 + elements;
      int64_t *out = output->data.i64;

      for (; in1 < in_end1; in1++, in2++, out++)
        *out = ((*in1 - *in2) * (*in1 - *in2));

      return kTfLiteOk;
    }
    default:
    {
      context->ReportError(context, "InputType is %d Unsupported", input1->type);
      return kTfLiteError;
    }
  }
}

} // namespace SquaredDifference
} // namespace custom
} // namespace tflite
} // namespace nnfw
