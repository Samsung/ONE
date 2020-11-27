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

#include "tflite/Diff.h"
#include "tflite/Quantization.h"
#include "tflite/interp/FunctionBuilder.h"

#include <iostream>
#include <cassert>

#include <chrono>
#include <random>

using namespace tflite;
using namespace nnfw::tflite;

TEST(NNAPI_Quickcheck_add_2, simple_test)
{
  int verbose = 0;
  int tolerance = 1;

  nnfw::misc::env::IntAccessor("VERBOSE").access(verbose);
  nnfw::misc::env::IntAccessor("TOLERANCE").access(tolerance);

  // Set random seed
  int SEED = std::chrono::system_clock::now().time_since_epoch().count();

  nnfw::misc::env::IntAccessor("SEED").access(SEED);

#define INT_VALUE(NAME, VALUE) IntVar NAME##_Value(#NAME, VALUE);
#include "add_2.lst"
#undef INT_VALUE

  const int32_t LEFT_N = LEFT_N_Value();
  const int32_t LEFT_C = LEFT_C_Value();
  const int32_t LEFT_H = LEFT_H_Value();
  const int32_t LEFT_W = LEFT_W_Value();

  const int32_t RIGHT_N = RIGHT_N_Value();
  const int32_t RIGHT_C = RIGHT_C_Value();
  const int32_t RIGHT_H = RIGHT_H_Value();
  const int32_t RIGHT_W = RIGHT_W_Value();

  const int32_t OFM_N = std::max(LEFT_N, RIGHT_N);
  const int32_t OFM_C = std::max(LEFT_C, RIGHT_C);
  const int32_t OFM_H = std::max(LEFT_H, RIGHT_H);
  const int32_t OFM_W = std::max(LEFT_W, RIGHT_W);

  // Initialize random number generator
  std::minstd_rand random(SEED);

  std::cout << "Configurations:" << std::endl;
#define PRINT_NEWLINE()     \
  {                         \
    std::cout << std::endl; \
  }
#define PRINT_VALUE(value)                                       \
  {                                                              \
    std::cout << "  " << #value << ": " << (value) << std::endl; \
  }
  PRINT_VALUE(SEED);
  PRINT_NEWLINE();

  PRINT_VALUE(LEFT_N);
  PRINT_VALUE(LEFT_C);
  PRINT_VALUE(LEFT_H);
  PRINT_VALUE(LEFT_W);
  PRINT_NEWLINE();

  PRINT_VALUE(RIGHT_N);
  PRINT_VALUE(RIGHT_C);
  PRINT_VALUE(RIGHT_H);
  PRINT_VALUE(RIGHT_W);
  PRINT_NEWLINE();

  PRINT_VALUE(OFM_N);
  PRINT_VALUE(OFM_C);
  PRINT_VALUE(OFM_H);
  PRINT_VALUE(OFM_W);
#undef PRINT_VALUE
#undef PRINT_NEWLINE

  // Configure left data
  const uint32_t left_size = LEFT_N * LEFT_C * LEFT_H * LEFT_W;
  float left_data[left_size] = {
    0.0f,
  };

  // Fill left data with random data
  {
    std::normal_distribution<float> left_dist(-1.0f, +1.0f);

    for (uint32_t off = 0; off < left_size; ++off)
    {
      left_data[off++] = left_dist(random);
    }
  }

  auto setup = [&](Interpreter &interp) {
    // Comment from 'context.h'
    //
    // Parameters for asymmetric quantization. Quantized values can be converted
    // back to float using:
    //    real_value = scale * (quantized_value - zero_point);
    //
    // Q: Is this necessary?
    TfLiteQuantizationParams quantization = make_default_quantization();

    // On AddTensors(N) call, T/F Lite interpreter creates N tensors whose index is [0 ~ N)
    interp.AddTensors(3);

    // Configure output
    interp.SetTensorParametersReadWrite(0, kTfLiteFloat32 /* type */, "output" /* name */,
                                        {OFM_N, OFM_H, OFM_W, OFM_C} /* dims */, quantization);

    // Configure input(s)
    interp.SetTensorParametersReadOnly(
      1, kTfLiteFloat32 /* type */, "left" /* name */, {LEFT_N, LEFT_H, LEFT_W, LEFT_C} /* dims */,
      quantization, reinterpret_cast<const char *>(left_data), left_size * sizeof(float));

    interp.SetTensorParametersReadWrite(2, kTfLiteFloat32 /* type */, "right" /* name */,
                                        {RIGHT_N, RIGHT_H, RIGHT_W, RIGHT_C} /* dims */,
                                        quantization);

    // Add Convolution Node
    //
    // NOTE AddNodeWithParameters take the ownership of param, and deallocate it with free
    //      So, param should be allocated with malloc
    auto param = make_alloc<TfLiteAddParams>();

    param->activation = kTfLiteActNone;

    // Run Add and store the result into Tensor #0
    //  - Read LHS from Tensor #1
    //  - Read RHS from Tensor #2,
    interp.AddNodeWithParameters({1, 2}, {0}, nullptr, 0, reinterpret_cast<void *>(param),
                                 BuiltinOpResolver().FindOp(BuiltinOperator_ADD, 1));

    interp.SetInputs({2});
    interp.SetOutputs({0});
  };

  const nnfw::tflite::FunctionBuilder builder(setup);

  RandomTestParam param;

  param.verbose = verbose;
  param.tolerance = tolerance;

  int res = RandomTestRunner{SEED, param}.run(builder);

  EXPECT_EQ(res, 0);
}
