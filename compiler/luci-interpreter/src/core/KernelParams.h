/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_CORE_KERNELPARAMS_H
#define LUCI_INTERPRETER_CORE_KERNELPARAMS_H

#include <luci/IR/AttrPadding.h>
#include <luci/IR/AttrFusedActFunc.h>
#include <luci_interpreter/core/DataType.h>
#include <luci_interpreter/core/Tensor.h>

#include <cstdint>
#include <functional>
#include <vector>

namespace luci_interpreter
{

// Inject commonly used types into `luci_interpreter` namespace for convenience.
using Activation = luci::FusedActFunc;
using Padding = luci::Padding;

struct AddParams
{
  Activation activation;
};

struct ArgMaxParams
{
  DataType output_type;
};

struct ConcatenationParams
{
  int axis;
  Activation activation;
};

struct Conv2DParams
{
  Padding padding;
  int32_t stride_height;
  int32_t stride_width;
  int32_t dilation_height_factor;
  int32_t dilation_width_factor;
  Activation activation;
};

struct DepthToSpaceParams
{
  int block_size;
};

struct DepthwiseConv2DParams
{
  Padding padding;
  int32_t depth_multiplier; // TODO Remove, as it can be calculated.
  int32_t stride_height;
  int32_t stride_width;
  int32_t dilation_height_factor;
  int32_t dilation_width_factor;
  Activation activation;
};

struct DivParams
{
  Activation activation;
};

struct FullyConnectedParams
{
  Activation activation;
};

struct InstanceNormParams
{
  float epsilon;
  Activation activation;
};

struct L2NormParams
{
  Activation activation;
};

struct LeakyReluParams
{
  float alpha;
};

struct LocalResponseNormalizationParams
{
  int32_t radius;
  float bias;
  float alpha;
  float beta;
};

struct MulParams
{
  Activation activation;
};

struct Pool2DParams
{
  Padding padding;
  int32_t filter_height;
  int32_t filter_width;
  int32_t stride_height;
  int32_t stride_width;
  Activation activation;
};

struct ReducerParams
{
  bool keep_dims;
};

struct ResizeBilinearParams
{
  bool align_corners;
  bool half_pixel_centers;
};

struct ResizeNearestNeighborParams
{
  bool align_corners;
  bool half_pixel_centers;
};

struct SubParams
{
  Activation activation;
};

struct SpaceToDepthParams
{
  int block_size;
};

struct SoftmaxParams
{
  float beta;
};

struct StridedSliceParams
{
  int32_t begin_mask;
  int32_t end_mask;
  int32_t ellipsis_mask;
  int32_t new_axis_mask;
  int32_t shrink_axis_mask;
};

struct SqueezeParams
{
  std::vector<int32_t> squeeze_dims;
};

struct TransposeConvParams
{
  Padding padding;
  int32_t stride_height;
  int32_t stride_width;
};

struct UnpackParams
{
  int axis;
};

// Compiled Func parameters

using CompiledFuncWrapper = int (*)(char *, void **);

struct ConfiguredCompiledFunc
{
  CompiledFuncWrapper wrapper;
  char *configuration;
};

using ConstructorCompiledFunc = ConfiguredCompiledFunc (*)(int ranks[], int *dims[]);

using DestructorCompiledFunc = void (*)(ConfiguredCompiledFunc *func);

struct CompiledParams
{
  ConstructorCompiledFunc constructor;
  DestructorCompiledFunc destructor;
  std::vector<luci_interpreter::Shape> input_shapes;
  std::vector<luci_interpreter::Shape> output_shapes;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_CORE_KERNELPARAMS_H
