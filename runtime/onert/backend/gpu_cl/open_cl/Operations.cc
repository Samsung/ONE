/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#include "Operations.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "absl/container/flat_hash_map.h"

#include "Shape.h"
#include "Status.h"
#include "InternalTensor.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

Padding2D &Padding2D::operator=(const Padding2D &value)
{
  prepended = value.prepended;
  appended = value.appended;
  return *this;
}

bool Padding2D::operator==(const Padding2D &value)
{
  return this->prepended == value.prepended && this->appended == value.appended;
}

bool Padding2D::operator!=(const Padding2D &value) { return !(*this == value); }

Padding2D &Padding2D::operator-(const Padding2D &value)
{
  prepended.h -= value.prepended.h;
  prepended.w -= value.prepended.w;
  appended.h -= value.appended.h;
  appended.w -= value.appended.w;
  return *this;
}

Padding3D &Padding3D::operator=(const Padding3D &value)
{
  prepended = value.prepended;
  appended = value.appended;
  return *this;
}

bool Padding3D::operator==(const Padding3D &value)
{
  return this->prepended == value.prepended && this->appended == value.appended;
}

bool Padding3D::operator!=(const Padding3D &value) { return !(*this == value); }

Padding3D &Padding3D::operator-(const Padding3D &value)
{
  prepended.h -= value.prepended.h;
  prepended.w -= value.prepended.w;
  prepended.d -= value.prepended.d;
  appended.h -= value.appended.h;
  appended.w -= value.appended.w;
  appended.d -= value.appended.d;
  return *this;
}

std::string ToString(enum OperationType op)
{
  switch (op)
  {
    case OperationType::ADD:
      return "add";
    case OperationType::UNKNOWN:
      return "unknown_operation";
  }
  return "";
}

OperationType OperationTypeFromString(const std::string &name)
{
  static const auto operations = new std::unordered_map<std::string, OperationType>({
    {"add", OperationType::ADD},
  });
  auto op = operations->find(name);
  return op == operations->end() ? OperationType::UNKNOWN : op->second;
}

namespace
{

template <typename T> T DivideRoundUp(T n, T divisor) { return (n - 1) / divisor + 1; }

int32_t CalculateOutputSizeBeforeStrides(int32_t input, int32_t kernel, int32_t padding,
                                         int32_t dilation)
{
  const int32_t dilated_kernel = (kernel - 1) * dilation + 1;
  return input + padding - dilated_kernel + 1;
}

template <Axis T>
int32_t CalculateOutputWithoutStrides(const BHWC &input, const Convolution2DAttributes &attr)
{
  return CalculateOutputSizeBeforeStrides(
    input.get<T>(), attr.weights.shape.get<T>(),
    attr.padding.prepended.get<T>() + attr.padding.appended.get<T>(), attr.dilations.get<T>());
}

template <Axis T>
int32_t CalculateOutputWithoutStrides(const BHWDC &input, const Convolution3DAttributes &attr)
{
  return CalculateOutputSizeBeforeStrides(
    input.get<T>(), attr.weights.shape.get<T>(),
    attr.padding.prepended.get<T>() + attr.padding.appended.get<T>(), attr.dilations.get<T>());
}

template <Axis T>
int32_t CalculateOutputWithoutStrides(const BHWC &input, const Pooling2DAttributes &attr)
{
  return CalculateOutputSizeBeforeStrides(input.get<T>(), attr.kernel.get<T>(),
                                          attr.padding.prepended.get<T>() +
                                            attr.padding.appended.get<T>(),
                                          /*dilation=*/1);
}

template <Axis T>
int32_t CalculateOutputWithoutStrides(const BHWDC &input, const Pooling3DAttributes &attr)
{
  return CalculateOutputSizeBeforeStrides(input.get<T>(), attr.kernel.get<T>(),
                                          attr.padding.prepended.get<T>() +
                                            attr.padding.appended.get<T>(),
                                          /*dilation=*/1);
}

template <Axis T>
int32_t CalculateOutput(const BHWC &input, const ConvolutionTransposedAttributes &attr)
{
  return (input.get<T>() - 1) * attr.stride.get<T>() -
         (attr.padding.prepended.get<T>() + attr.padding.appended.get<T>()) +
         attr.weights.shape.get<T>() + attr.adjacent.get<T>();
}

template <Axis T>
int32_t CalculateOutput(const BHWDC &input, const ConvolutionTransposed3DAttributes &attr)
{
  return (input.get<T>() - 1) * attr.stride.get<T>() -
         (attr.padding.prepended.get<T>() + attr.padding.appended.get<T>()) +
         attr.weights.shape.get<T>();
}

inline int32_t StridedSize(int32_t size, int32_t stride)
{
  return stride == 0 ? -1 : DivideRoundUp(size, stride);
}

template <Axis AxisT, typename AttrT> int32_t CalculateOutput(const BHWC &input, const AttrT &attr)
{
  return StridedSize(CalculateOutputWithoutStrides<AxisT>(input, attr),
                     attr.strides.template get<AxisT>());
}

template <Axis AxisT, typename AttrT> int32_t CalculateOutput(const BHWDC &input, const AttrT &attr)
{
  return StridedSize(CalculateOutputWithoutStrides<AxisT>(input, attr),
                     attr.strides.template get<AxisT>());
}

} // namespace

BHWC CalculateOutputShape(const BHWC &input, const MaxUnpooling2DAttributes &attr)
{
  return BHWC(
    input.b, input.h * attr.strides.h - attr.padding.prepended.h - attr.padding.appended.h,
    input.w * attr.strides.w - attr.padding.prepended.w - attr.padding.appended.w, input.c);
}

BHWDC CalculateOutputShape(const BHWDC &input, const MaxUnpooling3DAttributes &attr)
{
  return BHWDC(
    input.b, input.h * attr.strides.h - attr.padding.prepended.h - attr.padding.appended.h,
    input.w * attr.strides.w - attr.padding.prepended.w - attr.padding.appended.w,
    input.d * attr.strides.d - attr.padding.prepended.d - attr.padding.appended.d, input.c);
}

BHWC CalculateOutputShape(const BHWC &input, const Pooling2DAttributes &attr)
{
  return BHWC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
              CalculateOutput<Axis::WIDTH>(input, attr), input.c);
}

BHWDC CalculateOutputShape(const BHWDC &input, const Pooling3DAttributes &attr)
{
  return BHWDC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
               CalculateOutput<Axis::WIDTH>(input, attr), CalculateOutput<Axis::DEPTH>(input, attr),
               input.c);
}

BHWC CalculateOutputShape(const BHWC &input, const Convolution2DAttributes &attr)
{
  return BHWC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
              CalculateOutput<Axis::WIDTH>(input, attr),
              attr.weights.shape.get<Axis::OUTPUT_CHANNELS>());
}

BHWDC CalculateOutputShape(const BHWDC &input, const Convolution3DAttributes &attr)
{
  return BHWDC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
               CalculateOutput<Axis::WIDTH>(input, attr), CalculateOutput<Axis::DEPTH>(input, attr),
               attr.weights.shape.get<Axis::OUTPUT_CHANNELS>());
}

BHWC CalculateOutputShape(const BHWC &input, const ConvolutionTransposedAttributes &attr)
{
  return BHWC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
              CalculateOutput<Axis::WIDTH>(input, attr),
              attr.weights.shape.get<Axis::OUTPUT_CHANNELS>());
}

BHWDC CalculateOutputShape(const BHWDC &input, const ConvolutionTransposed3DAttributes &attr)
{
  return BHWDC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
               CalculateOutput<Axis::WIDTH>(input, attr), CalculateOutput<Axis::DEPTH>(input, attr),
               attr.weights.shape.get<Axis::OUTPUT_CHANNELS>());
}

BHWC CalculateOutputShape(const BHWC &input, const DepthwiseConvolution2DAttributes &attr)
{
  return BHWC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
              CalculateOutput<Axis::WIDTH>(input, attr),
              attr.weights.shape.get<Axis::OUTPUT_CHANNELS>() *
                attr.weights.shape.get<Axis::INPUT_CHANNELS>());
}

BHWDC CalculateOutputShape(const BHWDC &input, const DepthwiseConvolution3DAttributes &attr)
{
  return BHWDC(input.b, CalculateOutput<Axis::HEIGHT>(input, attr),
               CalculateOutput<Axis::WIDTH>(input, attr), CalculateOutput<Axis::DEPTH>(input, attr),
               attr.weights.shape.get<Axis::OUTPUT_CHANNELS>() *
                 attr.weights.shape.get<Axis::INPUT_CHANNELS>());
}

BHWC CalculateOutputShape(const BHWC &input, const SliceAttributes &attr)
{
  (void)input;
  return BHWC(StridedSize(attr.ends.b - attr.starts.b, attr.strides.b),
              StridedSize(attr.ends.h - attr.starts.h, attr.strides.h),
              StridedSize(attr.ends.w - attr.starts.w, attr.strides.w),
              StridedSize(attr.ends.c - attr.starts.c, attr.strides.c));
}

BHWDC CalculateOutputShape(const BHWDC &input, const Slice3DAttributes &attr)
{
  (void)input;
  return BHWDC(StridedSize(attr.ends.b - attr.starts.b, attr.strides.b),
               StridedSize(attr.ends.h - attr.starts.h, attr.strides.h),
               StridedSize(attr.ends.w - attr.starts.w, attr.strides.w),
               StridedSize(attr.ends.d - attr.starts.d, attr.strides.d),
               StridedSize(attr.ends.c - attr.starts.c, attr.strides.c));
}

BHWC CalculateOutputShape(const BHWC &input, const PadAttributes &attr)
{
  return BHWC(
    attr.appended.b + attr.prepended.b + input.b, attr.appended.h + attr.prepended.h + input.h,
    attr.appended.w + attr.prepended.w + input.w, attr.appended.c + attr.prepended.c + input.c);
}

BHWDC CalculateOutputShape(const BHWDC &input, const Pad3DAttributes &attr)
{
  return BHWDC(
    attr.appended.b + attr.prepended.b + input.b, attr.appended.h + attr.prepended.h + input.h,
    attr.appended.w + attr.prepended.w + input.w, attr.appended.d + attr.prepended.d + input.d,
    attr.appended.c + attr.prepended.c + input.c);
}

BHWC CalculateOutputShape(const BHWC &input, const FullyConnectedAttributes &attr)
{
  return BHWC(input.b, 1, 1, attr.weights.shape.o);
}

BHWC CalculateOutputShape(const BHWC &input, const MeanAttributes &attr)
{
  const int b = attr.dims.find(Axis::BATCH) == attr.dims.end() ? input.b : 1;
  const int h = attr.dims.find(Axis::HEIGHT) == attr.dims.end() ? input.h : 1;
  const int w = attr.dims.find(Axis::WIDTH) == attr.dims.end() ? input.w : 1;
  const int c = attr.dims.find(Axis::CHANNELS) == attr.dims.end() ? input.c : 1;
  return BHWC(b, h, w, c);
}

absl::Status CalculateOutputShape(const std::vector<BHWC> &input, const ConcatAttributes &attr,
                                  BHWC *output_shape)
{
  BHWC new_shape = input[0];
  switch (attr.axis)
  {
    case Axis::CHANNELS:
      for (size_t i = 1; i < input.size(); i++)
      {
        if (input[i].h != new_shape.h || input[i].w != new_shape.w || input[i].b != new_shape.b)
        {
          return absl::InvalidArgumentError(
            "Height, Width and Batch must be the same when concatenating "
            "by channels axis");
        }
        new_shape.c += input[i].c;
      }
      break;
    case Axis::HEIGHT:
      for (size_t i = 1; i < input.size(); i++)
      {
        if (input[i].w != new_shape.w || input[i].c != new_shape.c || input[i].b != new_shape.b)
        {
          return absl::InvalidArgumentError(
            "Channels, Width and Batch must be the same when concatenating "
            "by height axis");
        }
        new_shape.h += input[i].h;
      }
      break;
    case Axis::WIDTH:
      for (size_t i = 1; i < input.size(); i++)
      {
        if (input[i].h != new_shape.h || input[i].c != new_shape.c || input[i].b != new_shape.b)
        {
          return absl::InvalidArgumentError(
            "Height, Channels and Batch must be the same when concatenating "
            "by width axis");
        }
        new_shape.w += input[i].w;
      }
      break;
    case Axis::BATCH:
      for (size_t i = 1; i < input.size(); i++)
      {
        if (input[i].h != new_shape.h || input[i].c != new_shape.c || input[i].w != new_shape.w)
        {
          return absl::InvalidArgumentError(
            "Width, Height and Channels must be the same when concatenating "
            "by batch axis");
        }
        new_shape.b += input[i].b;
      }
      break;
    default:
      return absl::InvalidArgumentError("Invalid axis");
      break;
  }
  *output_shape = new_shape;
  return absl::OkStatus();
}

absl::Status CalculateOutputShape(const std::vector<BHWDC> &input, const ConcatAttributes &attr,
                                  BHWDC *output_shape)
{
  BHWDC new_shape = input[0];
  switch (attr.axis)
  {
    case Axis::CHANNELS:
      for (size_t i = 1; i < input.size(); ++i)
      {
        if (input[i].h != new_shape.h || input[i].w != new_shape.w || input[i].d != new_shape.d ||
            input[i].b != new_shape.b)
        {
          return absl::InvalidArgumentError("Height, Width, Batch and Depth must be the same when "
                                            "concatenating "
                                            "by channels axis");
        }
        new_shape.c += input[i].c;
      }
      break;
    case Axis::HEIGHT:
      for (size_t i = 1; i < input.size(); ++i)
      {
        if (input[i].w != new_shape.w || input[i].c != new_shape.c || input[i].d != new_shape.d ||
            input[i].b != new_shape.b)
        {
          return absl::InvalidArgumentError(
            "Width, Depth, Batch and Channels must be the same when "
            "concatenating "
            "by height axis");
        }
        new_shape.h += input[i].h;
      }
      break;
    case Axis::WIDTH:
      for (size_t i = 1; i < input.size(); ++i)
      {
        if (input[i].h != new_shape.h || input[i].c != new_shape.c || input[i].d != new_shape.d ||
            input[i].b != new_shape.b)
        {
          return absl::InvalidArgumentError(
            "Height, Depth, Batch and Channels must be the same when "
            "concatenating "
            "by width axis");
        }
        new_shape.w += input[i].w;
      }
      break;
    case Axis::DEPTH:
      for (size_t i = 1; i < input.size(); ++i)
      {
        if (input[i].w != new_shape.w || input[i].h != new_shape.h || input[i].c != new_shape.c ||
            input[i].b != new_shape.b)
        {
          return absl::InvalidArgumentError(
            "Width, Height, Batch and Channels must be the same when "
            "concatenating "
            "by depth axis");
        }
        new_shape.d += input[i].d;
      }
      break;
    case Axis::BATCH:
      for (size_t i = 1; i < input.size(); ++i)
      {
        if (input[i].w != new_shape.w || input[i].h != new_shape.h || input[i].c != new_shape.c ||
            input[i].d != new_shape.d)
        {
          return absl::InvalidArgumentError(
            "Width, Height, Depth and Channels must be the same when "
            "concatenating "
            "by batch axis");
        }
        new_shape.b += input[i].b;
      }
      break;
    default:
      return absl::InvalidArgumentError("Invalid axis");
  }
  *output_shape = new_shape;
  return absl::OkStatus();
}

float CalculateResizeScale(int32_t input_size, int32_t output_size, const Resize2DAttributes &attr)
{
  return attr.align_corners && input_size > 1 && output_size > 1
           ? static_cast<float>(input_size - 1) / (output_size - 1)
           : static_cast<float>(input_size) / output_size;
}

float CalculateResizeScale(int32_t input_size, int32_t output_size, const Resize3DAttributes &attr)
{
  return attr.align_corners && input_size > 1 && output_size > 1
           ? static_cast<float>(input_size - 1) / (output_size - 1)
           : static_cast<float>(input_size) / output_size;
}

BHWC CalculateOutputShape(const BHWC &input, const Resize2DAttributes &attr)
{
  return BHWC(input.b, attr.new_shape.h, attr.new_shape.w, input.c);
}

BHWDC CalculateOutputShape(const BHWDC &input, const Resize3DAttributes &attr)
{
  return BHWDC(input.b, attr.new_shape.h, attr.new_shape.w, attr.new_shape.d, input.c);
}

BHWC CalculateOutputShape(const BHWC &input, const TransposeAttributes &attr)
{
  return BHWC(input.get(attr.perm.b), input.get(attr.perm.h), input.get(attr.perm.w),
              input.get(attr.perm.c));
}

BHWDC CalculateOutputShape(const BHWDC &input, const Transpose3DAttributes &attr)
{
  return BHWDC(input.get(attr.perm.b), input.get(attr.perm.h), input.get(attr.perm.w),
               input.get(attr.perm.d), input.get(attr.perm.c));
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
