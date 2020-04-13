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

#include "gtest/gtest.h"

#include "tflite/ext/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/builtin_op_data.h"

#include "env.h"
#include "memory.h"
#include "misc/environment.h"
#include "misc/feature/Shape.h"

#include "tflite/Diff.h"
#include "tflite/Quantization.h"
#include "tflite/interp/FunctionBuilder.h"

#include <chrono>
#include <random>
#include <iostream>
#include <cassert>

using namespace tflite;
using namespace nnfw::tflite;

TEST(NNAPI_Quickcheck_softmax_1, simple_test)
{
  int verbose = 0;
  int tolerance = 1;

  nnfw::misc::env::IntAccessor("VERBOSE").access(verbose);
  nnfw::misc::env::IntAccessor("TOLERANCE").access(tolerance);

#define INT_VALUE(NAME, VALUE) IntVar NAME##_Value(#NAME, VALUE);
#include "softmax_quan_1.lst"
#undef INT_VALUE

  const int32_t IFM_C = 1;
  const int32_t IFM_H = IFM_H_Value();
  const int32_t IFM_W = IFM_W_Value();

  // Set random seed
  int SEED = std::chrono::system_clock::now().time_since_epoch().count();

  nnfw::misc::env::IntAccessor("SEED").access(SEED);

  // Initialize random number generator
  std::minstd_rand random(SEED);

  const nnfw::misc::feature::Shape ifm_shape{IFM_C, IFM_H, IFM_W};

  const int32_t OFM_C = IFM_C;
  const int32_t OFM_H = IFM_H;
  const int32_t OFM_W = IFM_W;

  auto setup = [&](Interpreter &interp) {
    // Comment from 'context.h'
    //
    // Parameters for asymmetric quantization. Quantized values can be converted
    // back to float using:
    //    real_value = scale * (quantized_value - zero_point);
    //
    // Q: Is this necessary?
    TfLiteQuantizationParams quantization;
    quantization.scale = 1.0f / 256;
    quantization.zero_point = 0;

    // On AddTensors(N) call, T/F Lite interpreter creates N tensors whose index is [0 ~ N)
    interp.AddTensors(2);

    // Configure Output Tensor
    interp.SetTensorParametersReadWrite(0, kTfLiteUInt8 /* type */, "output" /* name */,
                                        {1, IFM_H * IFM_W} /* dims */, quantization);

    // Configure Input Tensor
    interp.SetTensorParametersReadWrite(1, kTfLiteUInt8 /* type */, "input" /* name */,
                                        {1, IFM_H * IFM_W} /* batch_size, input_size */,
                                        quantization);

    // Add Softmax Node
    //
    // NOTE AddNodeWithParameters take the ownership of param, and deallocate it with free
    //      So, param should be allocated with malloc
    auto param = make_alloc<TfLiteSoftmaxParams>();

    param->beta = 1.0;

    // Run Softmax and store its result into Tensor #0
    //  - Read IFM from Tensor #1
    interp.AddNodeWithParameters({1}, {0}, nullptr, 0, reinterpret_cast<void *>(param),
                                 BuiltinOpResolver().FindOp(BuiltinOperator_SOFTMAX, 1));

    // Set Tensor #1 as Input #0, and Tensor #0 as Output #0
    interp.SetInputs({1});
    interp.SetOutputs({0});
  };

  const nnfw::tflite::FunctionBuilder builder(setup);

  RandomTestParam param;

  param.verbose = verbose;
  param.tolerance = tolerance;

  int res = RandomTestRunner{SEED, param}.run(builder);

  EXPECT_EQ(res, 0);
}
