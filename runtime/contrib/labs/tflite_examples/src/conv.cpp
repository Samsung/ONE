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

#include "tflite/ext/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/builtin_op_data.h"

#include <iostream>

using namespace tflite;
using namespace nnfw::tflite;

namespace vector
{

template <typename T> struct View
{
  virtual ~View() = default;

  virtual int32_t size(void) const = 0;
  virtual T at(uint32_t off) const = 0;
};
} // namespace vector

namespace feature
{

struct Shape
{
  int32_t C;
  int32_t H;
  int32_t W;
};

template <typename T> struct View
{
  virtual ~View() = default;

  virtual const Shape &shape(void) const = 0;
  virtual T at(uint32_t ch, uint32_t row, uint32_t col) const = 0;
};
} // namespace feature

namespace kernel
{

struct Shape
{
  int32_t N;
  int32_t C;
  int32_t H;
  int32_t W;
};

template <typename T> struct View
{
  virtual ~View() = default;

  virtual const Shape &shape(void) const = 0;
  virtual T at(uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) const = 0;
};
} // namespace kernel

const int32_t N = 1;
const int32_t C = 2;

class SampleBiasObject final : public vector::View<float>
{
public:
  SampleBiasObject() : _size(N)
  {
    // DO NOTHING
  }

public:
  int32_t size(void) const override { return _size; }

  float at(uint32_t off) const override { return 0.0f; }

private:
  int32_t _size;
};

class SampleFeatureObject final : public feature::View<float>
{
public:
  SampleFeatureObject()
  {
    _shape.C = C;
    _shape.H = 3;
    _shape.W = 4;

    const uint32_t size = _shape.C * _shape.H * _shape.W;

    for (uint32_t off = 0; off < size; ++off)
    {
      _value.emplace_back(off);
    }

    assert(_value.size() == size);
  }

public:
  const feature::Shape &shape(void) const override { return _shape; };

  float at(uint32_t ch, uint32_t row, uint32_t col) const override
  {
    return _value.at(ch * _shape.H * _shape.W + row * _shape.W + col);
  }

public:
  float &at(uint32_t ch, uint32_t row, uint32_t col)
  {
    return _value.at(ch * _shape.H * _shape.W + row * _shape.W + col);
  }

private:
  feature::Shape _shape;
  std::vector<float> _value;
};

class SampleKernelObject final : public kernel::View<float>
{
public:
  SampleKernelObject()
  {
    _shape.N = N;
    _shape.C = C;
    _shape.H = 3;
    _shape.W = 4;

    const uint32_t size = _shape.N * _shape.C * _shape.H * _shape.W;

    for (uint32_t off = 0; off < size; ++off)
    {
      _value.emplace_back(off);
    }

    assert(_value.size() == size);
  }

public:
  const kernel::Shape &shape(void) const override { return _shape; };

  float at(uint32_t nth, uint32_t ch, uint32_t row, uint32_t col) const override
  {
    return _value.at(nth * _shape.C * _shape.H * _shape.W + ch * _shape.H * _shape.W +
                     row * _shape.W + col);
  }

private:
  kernel::Shape _shape;
  std::vector<float> _value;
};

int main(int argc, char **argv)
{
  const SampleFeatureObject ifm;
  const SampleKernelObject kernel;
  const SampleBiasObject bias;

  const int32_t IFM_C = ifm.shape().C;
  const int32_t IFM_H = ifm.shape().H;
  const int32_t IFM_W = ifm.shape().W;

  const int32_t KER_N = kernel.shape().N;
  const int32_t KER_C = kernel.shape().C;
  const int32_t KER_H = kernel.shape().H;
  const int32_t KER_W = kernel.shape().W;

  const int32_t OFM_C = kernel.shape().N;
  const int32_t OFM_H = (IFM_H - KER_H) + 1;
  const int32_t OFM_W = (IFM_W - KER_W) + 1;

  // Assumption on this example
  assert(IFM_C == KER_C);
  assert(KER_N == bias.size());

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

  Interpreter interp;

  // On AddTensors(N) call, T/F Lite interpreter creates N tensors whose index is [0 ~ N)
  interp.AddTensors(5);

  // Configure OFM
  interp.SetTensorParametersReadWrite(0, kTfLiteFloat32 /* type */, "output" /* name */,
                                      {1 /*N*/, OFM_H, OFM_W, OFM_C} /* dims */, quantization);

  // Configure IFM
  interp.SetTensorParametersReadWrite(1, kTfLiteFloat32 /* type */, "input" /* name */,
                                      {1 /*N*/, IFM_H, IFM_W, IFM_C} /* dims */, quantization);

  // Configure Filter
  const uint32_t kernel_size = KER_N * KER_C * KER_H * KER_W;
  float kernel_data[kernel_size] = {
    0.0f,
  };

  // Fill kernel data in NHWC order
  {
    uint32_t off = 0;

    for (uint32_t nth = 0; nth < KER_N; ++nth)
    {
      for (uint32_t row = 0; row < KER_H; ++row)
      {
        for (uint32_t col = 0; col < KER_W; ++col)
        {
          for (uint32_t ch = 0; ch < KER_C; ++ch)
          {
            const auto value = kernel.at(nth, ch, row, col);
            kernel_data[off++] = value;
          }
        }
      }
    }

    assert(kernel_size == off);
  }

  interp.SetTensorParametersReadOnly(
    2, kTfLiteFloat32 /* type */, "filter" /* name */, {KER_N, KER_H, KER_W, KER_C} /* dims */,
    quantization, reinterpret_cast<const char *>(kernel_data), sizeof(kernel_data));

  // Configure Bias
  const uint32_t bias_size = bias.size();
  float bias_data[bias_size] = {
    0.0f,
  };

  // Fill bias data
  for (uint32_t off = 0; off < bias.size(); ++off)
  {
    bias_data[off] = bias.at(off);
  }

  interp.SetTensorParametersReadOnly(3, kTfLiteFloat32 /* type */, "bias" /* name */,
                                     {bias.size()} /* dims */, quantization,
                                     reinterpret_cast<const char *>(bias_data), sizeof(bias_data));

  // Add Convolution Node
  //
  // NOTE AddNodeWithParameters take the ownership of param, and deallocate it with free
  //      So, param should be allocated with malloc
  TfLiteConvParams *param = reinterpret_cast<TfLiteConvParams *>(malloc(sizeof(TfLiteConvParams)));

  param->padding = kTfLitePaddingValid;
  param->stride_width = 1;
  param->stride_height = 1;
  param->activation = kTfLiteActRelu;

  // Run Convolution and store its result into Tensor #0
  //  - Read IFM from Tensor #1
  //  - Read Filter from Tensor #2,
  //  - Read Bias from Tensor #3
  interp.AddNodeWithParameters({1, 2, 3}, {0}, nullptr, 0, reinterpret_cast<void *>(param),
                               BuiltinOpResolver().FindOp(BuiltinOperator_CONV_2D, 1));

  // Set Tensor #1 as Input #0, and Tensor #0 as Output #0
  interp.SetInputs({1});
  interp.SetOutputs({0});

  // Let's use NNAPI (if possible)
  interp.UseNNAPI(true);

  // Allocate Tensor
  interp.AllocateTensors();

  // Fill IFM data in HWC order
  {
    uint32_t off = 0;

    for (uint32_t row = 0; row < ifm.shape().H; ++row)
    {
      for (uint32_t col = 0; col < ifm.shape().W; ++col)
      {
        for (uint32_t ch = 0; ch < ifm.shape().C; ++ch)
        {
          const auto value = ifm.at(ch, row, col);
          interp.typed_input_tensor<float>(0)[off++] = value;
        }
      }
    }
  }

  // Let's Rock-n-Roll!
  interp.Invoke();

  // Print OFM
  {
    uint32_t off = 0;

    for (uint32_t row = 0; row < OFM_H; ++row)
    {
      for (uint32_t col = 0; col < OFM_W; ++col)
      {
        for (uint32_t ch = 0; ch < kernel.shape().N; ++ch)
        {
          std::cout << interp.typed_output_tensor<float>(0)[off++] << std::endl;
        }
      }
    }
  }

  return 0;
}
