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
#include "tflite/interp/FunctionBuilder.h"

#include <iostream>
#include <cassert>

#include <chrono>
#include <random>

using namespace tflite;
using namespace nnfw::tflite;

TEST(NNAPI_Quickcheck_mul_1, simple_test)
{
  int verbose = 0;
  int tolerance = 1;

  nnfw::misc::env::IntAccessor("VERBOSE").access(verbose);
  nnfw::misc::env::IntAccessor("TOLERANCE").access(tolerance);

  // Set random seed
  int SEED = std::chrono::system_clock::now().time_since_epoch().count();

  nnfw::misc::env::IntAccessor("SEED").access(SEED);

#define INT_VALUE(NAME, VALUE) IntVar NAME##_Value(#NAME, VALUE);
#include "mul_1.lst"
#undef INT_VALUE

  const int32_t LEFT_1D = LEFT_1D_Value();
  const int32_t LEFT_2D = LEFT_2D_Value();
  const int32_t LEFT_3D = LEFT_3D_Value();

  const int32_t RIGHT_W = RIGHT_W_Value();

  const int32_t OFM_1D = LEFT_1D_Value();
  const int32_t OFM_2D = LEFT_2D_Value();
  const int32_t OFM_3D = LEFT_3D_Value();

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

  PRINT_VALUE(LEFT_1D);
  PRINT_VALUE(LEFT_2D);
  PRINT_VALUE(LEFT_3D);
  PRINT_NEWLINE();

  PRINT_VALUE(RIGHT_W);
  PRINT_NEWLINE();

  PRINT_VALUE(OFM_1D);
  PRINT_VALUE(OFM_2D);
  PRINT_VALUE(OFM_3D);
#undef PRINT_VALUE
#undef PRINT_NEWLINE

  auto setup = [&](Interpreter &interp) {
    // Comment from 'context.h'
    //
    // Parameters for asymmetric quantization. Quantized values can be converted
    // back to float using:
    //    real_value = scale * (quantized_value - zero_point);
    //
    // Q: Is this necessary?
    TfLiteQuantizationParams quantization;
    quantization.zero_point = 0;

    // On AddTensors(N) call, T/F Lite interpreter creates N tensors whose index is [0 ~ N)
    interp.AddTensors(3);

    // Configure output
    float max_scale =
      std::numeric_limits<uint8_t>::max(); // * input1_scale(1.0f) * input2_scale(1.0f)
    quantization.scale = max_scale;
    interp.SetTensorParametersReadWrite(0, kTfLiteUInt8 /* type */, "output" /* name */,
                                        {OFM_1D, OFM_2D, OFM_3D} /* dims */, quantization);

    // Configure input(s)
    quantization.scale = 1.0f;
    interp.SetTensorParametersReadWrite(1, kTfLiteUInt8 /* type */, "left" /* name */,
                                        {LEFT_1D, LEFT_2D, LEFT_3D} /* dims */, quantization);

    interp.SetTensorParametersReadWrite(2, kTfLiteUInt8 /* type */, "right" /* name */,
                                        {RIGHT_W} /* dims */, quantization);

    // Add MUL Node
    //
    // NOTE AddNodeWithParameters take the ownership of param, and deallocate it with free
    //      So, param should be allocated with malloc
    auto param = make_alloc<TfLiteAddParams>();

    param->activation = kTfLiteActNone;

    // Run MUL and store the result into Tensor #0
    //  - Read Left from Tensor #1
    //  - Read Right from Tensor #2,
    interp.AddNodeWithParameters({1, 2}, {0}, nullptr, 0, reinterpret_cast<void *>(param),
                                 BuiltinOpResolver().FindOp(BuiltinOperator_MUL, 1));

    interp.SetInputs({1, 2});
    interp.SetOutputs({0});
  };

  const nnfw::tflite::FunctionBuilder builder(setup);

  RandomTestParam param;

  param.verbose = verbose;
  param.tolerance = tolerance;

  int res = RandomTestRunner{SEED, param}.run(builder);

  EXPECT_EQ(res, 0);
}
