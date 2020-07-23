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

#include "ConvBackend.h"

#include <nncc/core/ADT/kernel/Overlay.h>
#include <nncc/core/ADT/kernel/NHWCLayout.h>

#include <tensorflow/contrib/lite/kernels/register.h>
#include <tensorflow/contrib/lite/model.h>
#include <tensorflow/contrib/lite/builtin_op_data.h>

#include <cstdlib>
#include <stdexcept>

using namespace ::tflite;
using namespace ::tflite::ops::builtin;

/**
 * @brief Allocate memory with malloc and return a typed pointer
 *
 * NOTE This function throws std::bac_alloc exception on allocation failure
 */
template <typename T> T *typed_malloc(void)
{
  if (auto res = reinterpret_cast<T *>(malloc(sizeof(T))))
  {
    return res;
  }
  throw std::bad_alloc{};
}

// Comment from 'context.h'
//
// Parameters for asymmetric quantization. Quantized values can be converted
// back to float using:
//    real_value = scale * (quantized_value - zero_point);
static inline TfLiteQuantizationParams make_default_quantization(void)
{
  return TfLiteQuantizationParams{1.0f, 0};
}

static inline std::vector<int> as_dims(const nncc::core::ADT::feature::Shape &shape)
{
  const int N = 1;
  const int C = static_cast<int>(shape.depth());
  const int H = static_cast<int>(shape.height());
  const int W = static_cast<int>(shape.width());

  return std::vector<int>{N, H, W, C};
}

static inline std::vector<int> as_dims(const nncc::core::ADT::kernel::Shape &shape)
{
  const int N = static_cast<int>(shape.count());
  const int C = static_cast<int>(shape.depth());
  const int H = static_cast<int>(shape.height());
  const int W = static_cast<int>(shape.width());

  return std::vector<int>{N, H, W, C};
}

ConvBackend::ConvBackend(const nnsuite::conv::Model &model)
    : _ifm_name{model.ifm_name()}, _ofm_name{model.ofm_name()}
{
  using nncc::core::ADT::kernel::NHWCLayout;
  using nncc::core::ADT::kernel::Overlay;

  using nncc::core::ADT::kernel::make_overlay;
  using nncc::core::ADT::kernel::num_elements;

  // Set kernel data
  const auto &ker_shape = model.ker_shape();

  _kernel.resize(num_elements(ker_shape));

  auto kernel_overlay = make_overlay<float, NHWCLayout>(ker_shape, _kernel.data());

  for (uint32_t n = 0; n < ker_shape.count(); ++n)
  {
    for (uint32_t ch = 0; ch < ker_shape.depth(); ++ch)
    {
      for (uint32_t row = 0; row < ker_shape.height(); ++row)
      {
        for (uint32_t col = 0; col < ker_shape.width(); ++col)
        {
          kernel_overlay.at(n, ch, row, col) = model.ker_data().at(n, ch, row, col);
        }
      }
    }
  }

  // Set bias data
  _bias.resize(ker_shape.count(), 0.0f);

  // Initialize interpreter
  auto quantization = make_default_quantization();

  // Create Tensors
  //  0 -> OFM
  //  1 -> IFM
  //  2 -> Kernel
  //  3 -> Bias
  _interp.AddTensors(4);

  _interp.SetTensorParametersReadWrite(0, kTfLiteFloat32 /* type */, _ofm_name.c_str(),
                                       as_dims(model.ofm_shape()), quantization);

  _interp.SetTensorParametersReadWrite(1, kTfLiteFloat32 /* type */, _ifm_name.c_str(),
                                       as_dims(model.ifm_shape()), quantization);

  _interp.SetTensorParametersReadOnly(
      2, kTfLiteFloat32 /* type */, "kernel" /* name */, as_dims(model.ker_shape()), quantization,
      reinterpret_cast<const char *>(_kernel.data()), _kernel.size() * sizeof(float));

  _interp.SetTensorParametersReadOnly(
      3, kTfLiteFloat32 /* type */, "bias" /* name */, {static_cast<int>(_bias.size())},
      quantization, reinterpret_cast<const char *>(_bias.data()), _bias.size() * sizeof(float));

  auto param = typed_malloc<TfLiteConvParams>();

  param->padding = kTfLitePaddingValid;
  param->stride_width = 1;
  param->stride_height = 1;
  param->activation = kTfLiteActNone;

  _interp.AddNodeWithParameters({1, 2, 3}, {0}, nullptr, 0, reinterpret_cast<void *>(param),
                                BuiltinOpResolver().FindOp(BuiltinOperator_CONV_2D));

  _interp.SetInputs({1});
  _interp.SetOutputs({0});
}
