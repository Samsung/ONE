/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONNX_OP_REGISTRATION_H__
#define __ONNX_OP_REGISTRATION_H__

#include "ONNXNodeConverterRegistry.h"

#include "Op/Abs.h"
#include "Op/Add.h"
#include "Op/AveragePool.h"
#include "Op/BatchNormalization.h"
#include "Op/Concat.h"
#include "Op/Constant.h"
#include "Op/Conv.h"
#include "Op/ConvTranspose.h"
#include "Op/Div.h"
#include "Op/Dropout.h"
#include "Op/Equal.h"
#include "Op/Expand.h"
#include "Op/Flatten.h"
#include "Op/Gather.h"
#include "Op/Greater.h"
#include "Op/Gemm.h"
#include "Op/GlobalAveragePool.h"
#include "Op/Identity.h"
#include "Op/Less.h"
#include "Op/MatMul.h"
#include "Op/Max.h"
#include "Op/MaxPool.h"
#include "Op/Mul.h"
#include "Op/Pad.h"
#include "Op/Reciprocal.h"
#include "Op/ReduceMean.h"
#include "Op/Relu.h"
#include "Op/Reshape.h"
#include "Op/Shape.h"
#include "Op/Sigmoid.h"
#include "Op/Softmax.h"
#include "Op/Sqrt.h"
#include "Op/Sub.h"
#include "Op/Sum.h"
#include "Op/Tanh.h"
#include "Op/Transpose.h"
#include "Op/Unsqueeze.h"
#include "Op/Upsample.h"

namespace mir_onnx
{

inline void registerSupportedOps()
{
  auto &registry = NodeConverterRegistry::getInstance();

#define REG_CONVERTER(name, version, function) registry.registerConverter(name, version, function)
#define REG(name, version) REG_CONVERTER(#name, version, convert##name##V##version)
#define UNSUPPORTED(name, version) REG_CONVERTER(#name, version, nullptr)

  REG(Abs, 1);
  REG(Abs, 6);
  UNSUPPORTED(Abs, firstUnknownOpset);

  REG(Add, 1);
  REG(Add, 6);
  REG(Add, 7);
  UNSUPPORTED(Add, firstUnknownOpset);

  REG(AveragePool, 1);
  REG(AveragePool, 7);
  REG(AveragePool, 10);
  UNSUPPORTED(AveragePool, 11);
  UNSUPPORTED(AveragePool, firstUnknownOpset);

  REG(BatchNormalization, 1);
  REG(BatchNormalization, 6);
  REG(BatchNormalization, 7);
  REG(BatchNormalization, 9);
  UNSUPPORTED(BatchNormalization, firstUnknownOpset);

  REG(Concat, 1);
  REG(Concat, 4);
  UNSUPPORTED(Concat, 11);
  UNSUPPORTED(Concat, firstUnknownOpset);

  REG(Constant, 1);
  REG(Constant, 9);
  REG(Constant, 11);
  UNSUPPORTED(Constant, 12);
  UNSUPPORTED(Constant, firstUnknownOpset);

  REG(Conv, 1);
  UNSUPPORTED(Conv, 11);
  UNSUPPORTED(Conv, firstUnknownOpset);

  REG(ConvTranspose, 1);
  UNSUPPORTED(ConvTranspose, 11);
  UNSUPPORTED(ConvTranspose, firstUnknownOpset);

  UNSUPPORTED(Div, 1);
  UNSUPPORTED(Div, 6);
  REG(Div, 7);
  UNSUPPORTED(Div, firstUnknownOpset);

  REG(Dropout, 1);
  REG(Dropout, 6);
  REG(Dropout, 7);
  REG(Dropout, 10);
  UNSUPPORTED(Dropout, 12);
  UNSUPPORTED(Dropout, firstUnknownOpset);

  UNSUPPORTED(Equal, 1);
  REG(Equal, 7);
  REG(Equal, 11);
  UNSUPPORTED(Equal, firstUnknownOpset);

  REG(Expand, 8);
  UNSUPPORTED(Expand, firstUnknownOpset);

  REG(Flatten, 1);
  REG(Flatten, 9);
  UNSUPPORTED(Flatten, 11);
  UNSUPPORTED(Flatten, firstUnknownOpset);

  REG(Gather, 1);
  UNSUPPORTED(Gather, 11);
  UNSUPPORTED(Gather, firstUnknownOpset);

  REG(Gemm, 1);
  REG(Gemm, 6);
  REG(Gemm, 7);
  REG(Gemm, 9);
  REG(Gemm, 11);
  UNSUPPORTED(Gemm, firstUnknownOpset);

  UNSUPPORTED(GlobalAveragePool, 1);
  REG(GlobalAveragePool, 2);
  UNSUPPORTED(GlobalAveragePool, firstUnknownOpset);

  UNSUPPORTED(Greater, 1);
  REG(Greater, 7);
  REG(Greater, 9);
  UNSUPPORTED(Greater, firstUnknownOpset);

  REG(Identity, 1);
  UNSUPPORTED(Identity, firstUnknownOpset);

  UNSUPPORTED(Less, 1);
  REG(Less, 7);
  REG(Less, 9);
  UNSUPPORTED(Less, firstUnknownOpset);

  REG(MatMul, 1);
  REG(MatMul, 9);
  UNSUPPORTED(MatMul, firstUnknownOpset);

  REG(Max, 1);
  REG(Max, 6);
  REG(Max, 8);
  UNSUPPORTED(Max, firstUnknownOpset);

  REG(MaxPool, 1);
  REG(MaxPool, 8);
  REG(MaxPool, 10);
  UNSUPPORTED(MaxPool, 11);
  UNSUPPORTED(MaxPool, 12);
  UNSUPPORTED(MaxPool, firstUnknownOpset);

  UNSUPPORTED(Mul, 1);
  UNSUPPORTED(Mul, 6);
  REG(Mul, 7);
  UNSUPPORTED(Mul, firstUnknownOpset);

  REG(Pad, 1);
  REG(Pad, 2);
  UNSUPPORTED(Pad, 11);
  UNSUPPORTED(Pad, firstUnknownOpset);

  REG(Reciprocal, 1);
  REG(Reciprocal, 6);
  UNSUPPORTED(Reciprocal, firstUnknownOpset);

  REG(ReduceMean, 1);
  UNSUPPORTED(ReduceMean, 11);
  UNSUPPORTED(ReduceMean, firstUnknownOpset);

  REG(Relu, 1);
  REG(Relu, 6);
  UNSUPPORTED(Relu, firstUnknownOpset);

  REG(Reshape, 1);
  REG(Reshape, 5);
  UNSUPPORTED(Reshape, firstUnknownOpset);

  REG(Shape, 1);
  UNSUPPORTED(Shape, firstUnknownOpset);

  REG(Sigmoid, 1);
  REG(Sigmoid, 6);
  UNSUPPORTED(Sigmoid, firstUnknownOpset);

  REG(Softmax, 1);
  // TODO SoftmaxV11 is mostly the same, needs a check though
  UNSUPPORTED(Softmax, firstUnknownOpset);

  REG(Sqrt, 1);
  REG(Sqrt, 6);
  UNSUPPORTED(Sqrt, firstUnknownOpset);

  REG(Sub, 1);
  REG(Sub, 6);
  REG(Sub, 7);
  UNSUPPORTED(Sub, firstUnknownOpset);

  UNSUPPORTED(Sum, 1);
  UNSUPPORTED(Sum, 6);
  REG(Sum, 8);
  UNSUPPORTED(Sum, firstUnknownOpset);

  REG(Tanh, 1);
  REG(Tanh, 6);
  UNSUPPORTED(Tanh, firstUnknownOpset);

  REG(Transpose, 1);
  UNSUPPORTED(Transpose, firstUnknownOpset);

  REG(Unsqueeze, 1);
  UNSUPPORTED(Unsqueeze, 11);
  UNSUPPORTED(Unsqueeze, firstUnknownOpset);

  // Upsample-1 is not mentioned in onnx master and was considered experimental at the time
  REG(Upsample, 1);
  REG(Upsample, 7);
  REG(Upsample, 9);
  UNSUPPORTED(Upsample, firstUnknownOpset);

#undef REG
#undef REG_CONVERTER
#undef UNSUPPORTED
}

} // namespace mir_onnx

#endif // __ONNX_OP_REGISTRATION_H__
