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

TEST(NNAPI_Quickcheck_resize_bilinear_1, simple_test)
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
#include "resize_bilinear_1.lst"
#undef INT_VALUE

  const int32_t IFM_C = IFM_C_Value();
  const int32_t IFM_H = IFM_H_Value();
  const int32_t IFM_W = IFM_W_Value();

  const int32_t OFM_C = IFM_C;
  const int32_t OFM_H = OFM_H_Value();
  const int32_t OFM_W = OFM_W_Value();

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

  PRINT_VALUE(IFM_C);
  PRINT_VALUE(IFM_H);
  PRINT_VALUE(IFM_W);
  PRINT_NEWLINE();

  PRINT_VALUE(OFM_C);
  PRINT_VALUE(OFM_H);
  PRINT_VALUE(OFM_W);
#undef PRINT_VALUE
#undef PRINT_NEWLINE

  int32_t size_data[2] = {OFM_H, OFM_W};

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
    interp.AddTensors(3);

    // Configure OFM
    interp.SetTensorParametersReadWrite(0, kTfLiteFloat32 /* type */, "output" /* name */,
                                        {1 /*N*/, OFM_H, OFM_W, OFM_C} /* dims */, quantization);

    // Configure IFM
    interp.SetTensorParametersReadWrite(1, kTfLiteFloat32 /* type */, "input" /* name */,
                                        {1 /*N*/, IFM_H, IFM_W, IFM_C} /* dims */, quantization);

    // Configure Size
    interp.SetTensorParametersReadOnly(
      2, kTfLiteInt32 /* type */, "size" /* name */, {2} /* dims */, quantization,
      reinterpret_cast<const char *>(size_data), 2 * sizeof(int32_t));

    // NOTE AddNodeWithParameters take the ownership of param, and deallocate it with free
    //      So, param should be allocated with malloc
    auto param = make_alloc<TfLiteResizeBilinearParams>();

    // NOTE What is this?
    param->align_corners = false;

    interp.AddNodeWithParameters({1, 2}, {0}, nullptr, 0, reinterpret_cast<void *>(param),
                                 BuiltinOpResolver().FindOp(BuiltinOperator_RESIZE_BILINEAR, 1));

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
