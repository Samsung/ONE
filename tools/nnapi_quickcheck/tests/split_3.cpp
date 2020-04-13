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

TEST(NNAPI_Quickcheck_split_3, simple_test)
{
  int verbose = 0;
  int tolerance = 1;

  nnfw::misc::env::IntAccessor("VERBOSE").access(verbose);
  nnfw::misc::env::IntAccessor("TOLERANCE").access(tolerance);

#define INT_VALUE(NAME, VALUE) IntVar NAME##_Value(#NAME, VALUE);
#include "split_3.lst"
#undef INT_VALUE

  const int32_t IFM_H = IFM_H_Value();
  const int32_t IFM_W = IFM_W_Value();
  const int32_t NUM_SPLIT = NUM_SPLIT_Value();
  const int32_t AXIS = AXIS_Value();

  // Set random seed
  int SEED = std::chrono::system_clock::now().time_since_epoch().count();

  nnfw::misc::env::IntAccessor("SEED").access(SEED);

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

  PRINT_VALUE(IFM_H);
  PRINT_VALUE(IFM_W);
  PRINT_VALUE(NUM_SPLIT);
  PRINT_VALUE(AXIS);
#undef PRINT_VALUE
#undef PRINT_NEWLINE

  const int32_t OFM_H = IFM_H;
  const int32_t OFM_W = IFM_W;
  const int32_t axis[1] = {AXIS};

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
    interp.AddTensors(NUM_SPLIT + 2);

    // Configure Input Tensor(s)
    interp.SetTensorParametersReadOnly(0, kTfLiteInt32 /* type */, "axis" /* name */,
                                       {1} /* dims */, quantization,
                                       reinterpret_cast<const char *>(axis), 1 * sizeof(int32_t));

    interp.SetTensorParametersReadWrite(1, kTfLiteFloat32 /* type */, "input" /* name */,
                                        {IFM_H, IFM_W} /* dims */, quantization);

    // Configure Output Tensor
    std::vector<int> ofm_indexes;

    for (uint32_t n = 0; n < NUM_SPLIT; ++n)
    {
      const auto ofm_index = 2 + n;

      interp.SetTensorParametersReadWrite(ofm_index, kTfLiteFloat32 /* type */, "output" /* name */,
                                          {OFM_H, OFM_W} /* dims */, quantization);

      ofm_indexes.emplace_back(ofm_index);
    }

    auto *param = reinterpret_cast<TfLiteSplitParams *>(malloc(sizeof(TfLiteSplitParams)));

    param->num_splits = NUM_SPLIT;

    // Add SPLIT Node
    // Run SPLIT and store its result into Tensor #0
    //  - Read axis and IFM from Tensor #0 and #1, respectively
    interp.AddNodeWithParameters({0, 1}, ofm_indexes, nullptr, 0, reinterpret_cast<void *>(param),
                                 BuiltinOpResolver().FindOp(BuiltinOperator_SPLIT, 1));

    // Set Tensor #1 as Input #0, and Tensor #2 ~ #NUM_SPLIT+1 as Output #0
    interp.SetInputs({1});
    interp.SetOutputs(ofm_indexes);
  };

  const nnfw::tflite::FunctionBuilder builder(setup);

  RandomTestParam param;

  param.verbose = verbose;
  param.tolerance = tolerance;

  int res = RandomTestRunner{SEED, param}.run(builder);

  EXPECT_EQ(res, 0);
}
