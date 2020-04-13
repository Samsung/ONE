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

TEST(NNAPI_Quickcheck_concat_1, simple_test)
{
  int verbose = 0;
  int tolerance = 1;

  nnfw::misc::env::IntAccessor("VERBOSE").access(verbose);
  nnfw::misc::env::IntAccessor("TOLERANCE").access(tolerance);

  // Set random seed
  int SEED = std::chrono::system_clock::now().time_since_epoch().count();

  nnfw::misc::env::IntAccessor("SEED").access(SEED);

#define INT_VALUE(NAME, VALUE) IntVar NAME##_Value(#NAME, VALUE);
#include "concat_quan_1.lst"
#undef INT_VALUE

  // TODO Allow users to set concat axis!
  const int32_t CONCAT_COUNT = CONCAT_COUNT_Value();

  const int32_t IFM_H = IFM_H_Value();
  const int32_t IFM_W = IFM_W_Value();

  int32_t OFM_C = 0;
  const int32_t OFM_H = IFM_H;
  const int32_t OFM_W = IFM_W;

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

  PRINT_VALUE(CONCAT_COUNT);
  PRINT_NEWLINE();

  PRINT_VALUE(IFM_H);
  PRINT_VALUE(IFM_W);
#undef PRINT_VALUE
#undef PRINT_NEWLINE

  // Randomize IFM depth
  std::default_random_engine generator(SEED);
  std::uniform_int_distribution<int> distribution(1, 8);

  std::vector<int32_t> depths;

  for (int32_t n = 0; n < CONCAT_COUNT; ++n)
  {
    const auto depth = distribution(generator);

    OFM_C += depth;
    depths.emplace_back(depth);
  }

  auto setup = [&](Interpreter &interp) {
    // Comment from 'context.h'
    //
    // Parameters for asymmetric quantization. Quantized values can be converted
    // back to float using:
    //    real_value = scale * (quantized_value - zero_point);
    //
    // Q: Is this necessary?
    TfLiteQuantizationParams quantization;
    quantization.scale = 1.0f;
    quantization.zero_point = 0;

    // On AddTensors(N) call, T/F Lite interpreter creates N tensors whose index is [0 ~ N)
    interp.AddTensors(depths.size() + 1);

    // Configure OFM
    interp.SetTensorParametersReadWrite(0, kTfLiteUInt8 /* type */, "output" /* name */,
                                        {1 /*N*/, OFM_H, OFM_W, OFM_C} /* dims */, quantization);

    // Configure IFM(s)
    std::vector<int> ifm_indexes;

    for (uint32_t n = 0; n < depths.size(); ++n)
    {
      const auto ifm_index = 1 + n;
      const auto IFM_C = depths.at(n);

      interp.SetTensorParametersReadWrite(ifm_index, kTfLiteUInt8 /* type */, "input" /* name */,
                                          {1 /*N*/, IFM_H, IFM_W, IFM_C} /* dims */, quantization);

      ifm_indexes.emplace_back(ifm_index);
    }

    // Add Concat Node
    //
    // NOTE AddNodeWithParameters take the ownership of param, and deallocate it with free
    //      So, param should be allocated with malloc
    auto param = make_alloc<TfLiteConcatenationParams>();

    param->activation = kTfLiteActNone;
    param->axis = 3;

    // Run Convolution and store its result into Tensor #0
    //  - Read IFM from Tensor #1
    interp.AddNodeWithParameters(ifm_indexes, {0}, nullptr, 0, reinterpret_cast<void *>(param),
                                 BuiltinOpResolver().FindOp(BuiltinOperator_CONCATENATION, 1));

    // Set Tensor #1 as Input #0, and Tensor #0 as Output #0
    interp.SetInputs(ifm_indexes);
    interp.SetOutputs({0});
  };

  const nnfw::tflite::FunctionBuilder builder(setup);

  RandomTestParam param;

  param.verbose = verbose;
  param.tolerance = tolerance;

  int res = RandomTestRunner{SEED, param}.run(builder);

  EXPECT_EQ(res, 0);
}
