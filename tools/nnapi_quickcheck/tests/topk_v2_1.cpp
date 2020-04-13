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

#include <chrono>
#include <iostream>

using namespace tflite;
using namespace nnfw::tflite;

TEST(NNAPI_Quickcheck_topk_v2_1, simple_test)
{
  // Set random seed
  int SEED = std::chrono::system_clock::now().time_since_epoch().count();

  nnfw::misc::env::IntAccessor("SEED").access(SEED);

  // Set random test parameters
  int verbose = 0;
  int tolerance = 1;

  nnfw::misc::env::IntAccessor("VERBOSE").access(verbose);
  nnfw::misc::env::IntAccessor("TOLERANCE").access(tolerance);

#define INT_VALUE(NAME, VALUE) IntVar NAME##_Value(#NAME, VALUE);
#include "topk_v2_1.lst"
#undef INT_VALUE

  const int32_t INPUT_DATA = INPUT_DATA_Value();
  const int32_t K = K_Value();

  const int32_t OUTPUT_VALUES = K;
  const int32_t OUTPUT_INDICES = K;

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

  PRINT_VALUE(INPUT_DATA);
  PRINT_VALUE(K);
  PRINT_NEWLINE();

  PRINT_VALUE(OUTPUT_VALUES);
  PRINT_VALUE(OUTPUT_INDICES);
#undef PRINT_VALUE
#undef PRINT_NEWLINE

  // Fill the K data
  int32_t k_data[1] = {K};

  auto setup = [&](Interpreter &interp) {
    // Comment from 'context.h'
    //
    // Parameters for asymmetric quantization. Quantized values can be converted
    // back to float using:
    //    real_value = scale * (quantized_value - zero_point);
    //
    // Q: Is this necessary?
    // A: This may be necessary, because quantization values(scale, zero_point) of TENSOR_INT32 and
    // TENSOR_QUANT8_ASYMM are passed on to the runtime.
    TfLiteQuantizationParams quantization = make_default_quantization();

    // On AddTensors(N) call, T/F Lite interpreter creates N tensors whose index is [0 ~ N)
    interp.AddTensors(4);

    // Configure INPUT_DATA
    interp.SetTensorParametersReadWrite(0, kTfLiteFloat32 /* type */, "input" /* name */,
                                        {INPUT_DATA} /* dims */, quantization);

    // Configure K
    interp.SetTensorParametersReadOnly(1, kTfLiteInt32 /* type */, "k" /* name */, {1} /* dims */,
                                       quantization, reinterpret_cast<const char *>(k_data),
                                       sizeof(k_data));

    // Configure OUTPUT_VALUES
    interp.SetTensorParametersReadWrite(2, kTfLiteFloat32 /* type */, "output_values" /* name */,
                                        {OUTPUT_VALUES} /* dims */, quantization);

    // Configure OUTPUT_INDICES
    interp.SetTensorParametersReadWrite(3, kTfLiteInt32 /* type */, "output_indices" /* name */,
                                        {OUTPUT_INDICES} /* dims */, quantization);

    // Add TopK_V2 Node
    // Run TopK_V2 and store its result into Tensor #2 and #3
    //  - Read input data and K from Tensor #0 and #1, respectively
    interp.AddNodeWithParameters({0, 1}, {2, 3}, nullptr, 0, nullptr,
                                 BuiltinOpResolver().FindOp(BuiltinOperator_TOPK_V2, 1));

    // Set Tensor #0 as Input, and Tensor #2 and #3 as Output
    interp.SetInputs({0});
    interp.SetOutputs({2, 3});
  };

  const nnfw::tflite::FunctionBuilder builder(setup);

  RandomTestParam param;

  param.verbose = verbose;
  param.tolerance = tolerance;

  int res = RandomTestRunner{SEED, param}.run(builder);

  EXPECT_EQ(res, 0);
}
