/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef LUCI_INTERPRETER_KERNELS_UTILS_H
#define LUCI_INTERPRETER_KERNELS_UTILS_H

#include "core/KernelParams.h"
#include "luci_interpreter/core/Tensor.h"

#include <tensorflow/lite/kernels/internal/types.h>

#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

#define LUCI_INTERPRETER_CHECK(cond)                                                         \
  if (!(cond))                                                                               \
    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + +"(" + \
                             std::string(#cond) + ") was not true.");

inline int32_t computePadding(int32_t stride, int32_t dilation_rate, int32_t in_size,
                              int32_t filter_size, int32_t out_size)
{
  const int32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  const int32_t padding = ((out_size - 1) * stride + effective_filter_size - in_size) / 2;
  return padding > 0 ? padding : 0;
}

inline int32_t computePaddingWithOffset(int32_t stride, int32_t dilation_rate, int32_t in_size,
                                        int32_t filter_size, int32_t out_size, int32_t *offset)
{
  int32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  int32_t total_padding = ((out_size - 1) * stride + effective_filter_size - in_size);
  total_padding = total_padding > 0 ? total_padding : 0;
  *offset = total_padding % 2;
  return total_padding / 2;
}

inline int32_t computeOutputSize(Padding padding, int32_t image_size, int32_t filter_size,
                                 int32_t stride, int32_t dilation_rate = 1)
{
  const int32_t effective_filter_size = (filter_size - 1) * dilation_rate + 1;
  switch (padding)
  {
    case Padding::SAME:
      return (image_size + stride - 1) / stride;
    case Padding::VALID:
      return (image_size + stride - effective_filter_size) / stride;
    default:
      assert(false);
      return 0;
  }
}

inline int32_t calcOffset(const Shape &shape, int32_t d0, int32_t d1, int32_t d2, int32_t d3)
{
  return ((d0 * shape.dim(1) + d1) * shape.dim(2) + d2) * shape.dim(3) + d3;
}

template <typename T>
void calculateActivationRange(Activation activation, T *activation_min, T *activation_max);

void calculateActivationRangeQuantized(Activation activation, const Tensor *output,
                                       int32_t *activation_min, int32_t *activation_max);

template <typename T> constexpr bool one_of_types() { return false; }

// Checks if T is equal to one of {U,Other} types
template <typename T, typename U, typename... Other> constexpr bool one_of_types()
{
  return std::is_same<T, U>::value || one_of_types<T, Other...>();
}

/**
 * Fills activation min and max parameters depending on given data type and activation
 *
 * T is a template parameter, so after optimization this code left with only required if case
 *
 * @tparam T data type of arithmetic operation output tensor
 * @param params tflite params to fill
 * @param activation luci_interpreter::Activation of arithmetic operation
 */
template <typename T>
void fillArithmeticActivationRange(tflite::ArithmeticParams &p, Activation act)
{
  static_assert(one_of_types<T, float, int32_t, int64_t>(), "Unsupported dtype");

  if (std::is_same<T, float>::value)
    calculateActivationRange(act, &p.float_activation_min, &p.float_activation_max);
  if (std::is_same<T, int32_t>::value)
    calculateActivationRange(act, &p.quantized_activation_min, &p.quantized_activation_max);
  else
    calculateActivationRange(act, &p.int64_activation_min, &p.int64_activation_max);
}

// Decompose a double multiplier into a Q0.31 int32 representation of its
// significand, and shift representation of its exponent.
//
// Handles an arbitrary positive multiplier. The 'shift' output-value is
// basically the 'floating-point exponent' of the multiplier:
// Negative for a right-shift (when the multiplier is <1), positive for a
// left-shift (when the multiplier is >1)
void quantizeMultiplier(double double_multiplier, int32_t *quantized_multiplier, int *shift);

// Decompose a double multiplier into a Q0.31 int32 representation of its
// significand, and shift representation of NEGATIVE its exponent ---
// this is intended as a RIGHT-shift.
//
// Restricted to the case where the multiplier < 1 (and non-negative).
void quantizeMultiplierSmallerThanOneExp(double double_multiplier, int32_t *quantized_multiplier,
                                         int *left_shift);

Shape calculateShapeForBroadcast(const Shape &input1_shape, const Shape &input2_shape);

inline double getQuantizedConvolutionMultipler(float input_scale, float filter_scale,
                                               float output_scale)
{
  const double input_product_scale = static_cast<double>(input_scale * filter_scale);
  LUCI_INTERPRETER_CHECK(input_product_scale >= 0);
  return input_product_scale / static_cast<double>(output_scale);
}

// TODO rename getQuantizedConvolutionMultiplers to something more general
// it is used for non conv operators too
inline std::vector<double> getQuantizedConvolutionMultiplers(float input_scale,
                                                             const std::vector<float> &filter_scale,
                                                             float output_scale)
{
  std::vector<double> effective_output_scales;
  size_t n = filter_scale.size();
  effective_output_scales.reserve(n);
  for (size_t i = 0; i < n; ++i)
  {
    effective_output_scales.push_back(
      getQuantizedConvolutionMultipler(input_scale, filter_scale[i], output_scale));
  }
  return effective_output_scales;
}

struct ChannelQuantMultipliers
{
  int shift;
  int32_t multiplier;
  ChannelQuantMultipliers() = default;
};

inline std::vector<ChannelQuantMultipliers>
quantizeMultipliers(const std::vector<double> &effective_scale)
{
  size_t n = effective_scale.size();
  std::vector<ChannelQuantMultipliers> params(n);
  for (size_t i = 0; i < n; ++i)
  {
    quantizeMultiplier(effective_scale[i], &params[i].multiplier, &params[i].shift);
  }
  return params;
}

// Helper wrapper to hide broadcast logic
template <typename T> class BroadcastableWrapper
{
public:
  BroadcastableWrapper(const std::vector<T> &v) : _v(v), _stride(v.size() == 1 ? 0 : 1) {}

  T operator[](int idx) { return _v[idx * _stride]; }

private:
  const std::vector<T> &_v;
  int _stride;
};

inline tflite::RuntimeShape getTensorShape(const Tensor *tensor)
{
  if (tensor == nullptr)
    return tflite::RuntimeShape();

  const Shape &shape = tensor->shape();
  tflite::RuntimeShape runtime_shape(shape.num_dims());
  for (int i = 0; i < shape.num_dims(); ++i)
  {
    runtime_shape.SetDim(i, shape.dim(i));
  }
  return runtime_shape;
}

template <typename T> const T *getTensorData(const Tensor *tensor)
{
  return tensor != nullptr ? tensor->data<T>() : nullptr;
}

template <typename T> T *getTensorData(Tensor *tensor)
{
  return tensor != nullptr ? tensor->data<T>() : nullptr;
}

// A list of tensors in a format that can be used by kernels like split and
// concatenation.
template <typename T, bool is_const> class VectorOfTensors
{
public:
  using ElementT = typename std::conditional<is_const, const T, T>::type;
  using TensorT = typename std::conditional<is_const, const Tensor, Tensor>::type;

  // Build with the tensors in 'tensor_list'.
  explicit VectorOfTensors(const std::vector<TensorT *> &tensor_list)
  {
    const int num_tensors = tensor_list.size();

    all_data_.reserve(num_tensors);
    all_shape_.reserve(num_tensors);
    all_shape_ptr_.reserve(num_tensors);

    for (TensorT *tensor : tensor_list)
    {
      all_data_.push_back(getTensorData<T>(tensor));
      all_shape_.push_back(getTensorShape(tensor));
    }

    // Taking the pointer from inside a std::vector is only OK if the vector is
    // never modified, so we populate all_shape in the previous loop and then we
    // are free to grab iterators here.
    for (tflite::RuntimeShape &shape : all_shape_)
    {
      all_shape_ptr_.push_back(&shape);
    }
  }
  // Return a pointer to the data pointers of all tensors in the list. For
  // example:
  //   float* const* f = v.data();
  //   f[0][1] is the second element of the first tensor.
  ElementT *const *data() const { return all_data_.data(); }

  // Return a pointer the shape pointers of all tensors in the list. For
  // example:
  //   const RuntimeShape* const* d = v.dims();
  //   dims[1] are the dimensions of the second tensor in the list.
  const tflite::RuntimeShape *const *shapes() const { return all_shape_ptr_.data(); }

private:
  std::vector<ElementT *> all_data_;
  std::vector<tflite::RuntimeShape> all_shape_;
  std::vector<tflite::RuntimeShape *> all_shape_ptr_;
};

// A list of quantized tensors in a format that can be used by kernels like
// split and concatenation.
template <bool is_const> class VectorOfQuantizedTensors : public VectorOfTensors<uint8_t, is_const>
{
public:
  using typename VectorOfTensors<uint8_t, is_const>::TensorT;

  // Build with the tensors in 'tensor_list'.
  explicit VectorOfQuantizedTensors(const std::vector<TensorT *> &tensor_list)
    : VectorOfTensors<uint8_t, is_const>(tensor_list)
  {
    for (TensorT *tensor : tensor_list)
    {
      zero_point_.push_back(tensor->zero_point());
      scale_.push_back(tensor->scale());
    }
  }

  const float *scale() const { return scale_.data(); }
  const int32_t *zero_point() const { return zero_point_.data(); }

private:
  std::vector<int32_t> zero_point_;
  std::vector<float> scale_;
};

} // namespace kernels
} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_KERNELS_UTILS_H
