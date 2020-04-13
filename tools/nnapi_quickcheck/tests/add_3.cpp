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
#include "tflite/TensorShapeUtils.h"
#include "tflite/interp/FunctionBuilder.h"

#include <iostream>
#include <cassert>

#include <chrono>
#include <random>

using namespace tflite;
using namespace nnfw::tflite;

TEST(NNAPI_Quickcheck_add_3, simple_test)
{
  // Set random seed
  int SEED = std::chrono::system_clock::now().time_since_epoch().count();

  nnfw::misc::env::IntAccessor("SEED").access(SEED);

  // Initialize random number generator
  std::minstd_rand random(SEED);

#define STR_VALUE(NAME, VALUE) StrVar NAME##_Value(#NAME, VALUE);
#include "add_3.lst"
#undef STR_VALUE

  const auto LHS_SHAPE = nnfw::misc::tensor::Shape::from(LHS_SHAPE_Value());
  const auto RHS_SHAPE = nnfw::misc::tensor::Shape::from(RHS_SHAPE_Value());
  const auto OUT_SHAPE = nnfw::tflite::broadcast(LHS_SHAPE, RHS_SHAPE);

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

  PRINT_VALUE(LHS_SHAPE);
  PRINT_VALUE(RHS_SHAPE);
  PRINT_VALUE(OUT_SHAPE);
#undef PRINT_VALUE
#undef PRINT_NEWLINE

  auto setup = [&](Interpreter &interp) {
    using nnfw::tflite::as_dims;

    // Comment from 'context.h'
    //
    // Parameters for asymmetric quantization. Quantized values can be converted
    // back to float using:
    //    real_value = scale * (quantized_value - zero_point);
    //
    // Q: Is this necessary?
    TfLiteQuantizationParams quantization;

    quantization.scale = 1;
    quantization.zero_point = 0;

    // On AddTensors(N) call, T/F Lite interpreter creates N tensors whose index is [0 ~ N)
    interp.AddTensors(3);

    // Configure output
    interp.SetTensorParametersReadWrite(0, kTfLiteFloat32 /* type */, "output" /* name */,
                                        as_dims(OUT_SHAPE), quantization);

    // Configure input(s)
    interp.SetTensorParametersReadWrite(1, kTfLiteFloat32 /* type */, "left" /* name */,
                                        as_dims(LHS_SHAPE), quantization);

    interp.SetTensorParametersReadWrite(2, kTfLiteFloat32 /* type */, "right" /* name */,
                                        as_dims(RHS_SHAPE), quantization);

    // Add Convolution Node
    //
    // NOTE AddNodeWithParameters take the ownership of param, and deallocate it with free
    //      So, param should be allocated with malloc
    auto param = make_alloc<TfLiteAddParams>();

    param->activation = kTfLiteActNone;

    // Run Add and store the result into Tensor #0
    //  - Read Left from Tensor #1
    //  - Read Left from Tensor #2,
    interp.AddNodeWithParameters({1, 2}, {0}, nullptr, 0, reinterpret_cast<void *>(param),
                                 BuiltinOpResolver().FindOp(BuiltinOperator_ADD, 1));

    interp.SetInputs({1, 2});
    interp.SetOutputs({0});
  };

  const nnfw::tflite::FunctionBuilder builder(setup);

  RandomTestParam param;

  param.verbose = 0;
  param.tolerance = 1;

  nnfw::misc::env::IntAccessor("VERBOSE").access(param.verbose);
  nnfw::misc::env::IntAccessor("TOLERANCE").access(param.tolerance);

  int res = RandomTestRunner{SEED, param}.run(builder);

  EXPECT_EQ(res, 0);
}
