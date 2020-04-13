/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "FullyConnected.h"
#include "Common.h"

#include "QuantizationHelpers.h"

#include "mir/Tensor.h"

namespace mir_interpreter
{

template <typename T>
static void fullyConnected2D(const mir::TensorVariant &input, const mir::TensorVariant &weights,
                             mir::TensorVariant &output)
{
  assert(input.getShape().rank() == 2);
  assert(weights.getShape().rank() == 2);
  assert(input.getShape().dim(1) == weights.getShape().dim(0));

  auto in_raw = reinterpret_cast<T *>(input.atOffset(0));
  auto weight_raw = reinterpret_cast<T *>(weights.atOffset(0));
  auto output_raw = reinterpret_cast<T *>(output.atOffset(0));

  auto rows = output.getShape().dim(0);
  auto cols = output.getShape().dim(1);
  auto N = input.getShape().dim(1);
  auto wcols = weights.getShape().dim(1);

  for (int32_t r = 0; r < rows; ++r)
  {
    for (int32_t k = 0; k < N; ++k)
    {
      auto in = in_raw[r * N + k];

      for (int32_t c = 0; c < cols; ++c)
      {
        output_raw[r * cols + c] += in * weight_raw[k * wcols + c];
      }
    }
  }
}

template <typename T> struct FullyConnectedImpl
{
  static void run(const mir::TensorVariant &inputv, const mir::TensorVariant &weightsv,
                  const mir::ops::FullyConnectedOp &op, mir::TensorVariant &res,
                  const mir::TensorVariant *biasv);
};

template <typename T>
void FullyConnectedImpl<T>::run(const mir::TensorVariant &inputv,
                                const mir::TensorVariant &weightsv,
                                const mir::ops::FullyConnectedOp &op, mir::TensorVariant &res,
                                const mir::TensorVariant *biasv)
{
  if (biasv)
  {
    throw std::runtime_error("non-quantized FullyConnected with fused bias is unsupported");
  }

  mir::Tensor<T> input{inputv};
  mir::Tensor<T> weights{weightsv};

  erase<T>(res);

  if (input.getShape().rank() == 2 && weights.getShape().rank() == 2 && res.getShape().rank() == 2)
  {
    // optimized case for 2d matrix multiplication
    fullyConnected2D<T>(inputv, weightsv, res);
    return;
  }

  mir::Tensor<T> accessor(res);

  const mir::Shape &in_shape = input.getShape();
  int32_t in_rank = in_shape.rank();

  const mir::Shape &w_shape = weights.getShape();
  int32_t w_rank = w_shape.rank();

  assert(in_shape.dim(in_rank - 1) == w_shape.dim(w_rank - 2));
  (void)in_rank;

  mir::ShapeRange out_range(res.getShape());

  int32_t len = w_shape.dim(w_rank - 2);

  for (auto &out_index : out_range)
  {
    mir::Index t_index = out_index;
    T &output_element = accessor.at(out_index);
    int32_t col = t_index.at(w_rank - 1);
    int32_t row = t_index.at(w_rank - 2);
    for (int32_t i = 0; i < len; ++i)
    {
      t_index.at(w_rank - 1) = i;
      T in = input.at(t_index);
      t_index.at(w_rank - 1) = col;
      t_index.at(w_rank - 2) = i;
      T w = weights.at(t_index);
      t_index.at(w_rank - 2) = row;
      output_element += in * w;
    }
  }
}

template <> struct FullyConnectedImpl<uint8_t>
{
  static void run(const mir::TensorVariant &inputv, const mir::TensorVariant &weightsv,
                  const mir::ops::FullyConnectedOp &op, mir::TensorVariant &res,
                  const mir::TensorVariant *biasv);
};

void FullyConnectedImpl<uint8_t>::run(const mir::TensorVariant &inputv,
                                      const mir::TensorVariant &weightsv,
                                      const mir::ops::FullyConnectedOp &op, mir::TensorVariant &res,
                                      const mir::TensorVariant *biasv)
{
  if (!biasv)
  {
    throw std::runtime_error{"Quantized FullyConnected cannot be executed without fused bias"};
  }

  const auto &input_type = inputv.getType();
  const auto &weights_type = weightsv.getType();
  const auto &bias_type = biasv->getType();
  const auto &output_type = op.getOutput(0)->getType();
  (void)bias_type;

  assert(input_type.isQuantized());
  assert(weights_type.isQuantized());
  assert(bias_type.isQuantized());
  assert(output_type.isQuantized());
  assert(input_type.getElementType() == mir::DataType::UINT8);
  assert(weights_type.getElementType() == mir::DataType::UINT8);
  assert(bias_type.getElementType() == mir::DataType::INT32);

  int32_t input_offset = -input_type.getQuantization().getZeroPoint();
  int32_t weights_offset = -weights_type.getQuantization().getZeroPoint();
  int32_t output_offset = output_type.getQuantization().getZeroPoint();

  double input_scale = input_type.getQuantization().getScale();
  double weights_scale = weights_type.getQuantization().getScale();
  double output_scale = output_type.getQuantization().getScale();

  double real_multiplier = input_scale * weights_scale / output_scale;
  int32_t output_multiplier = 0;
  int output_shift = 0;
  QuantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);

  const mir::Shape &in_shape = inputv.getShape();
  const mir::Shape &weights_shape = weightsv.getShape();
  const mir::Shape &out_shape = op.getOutputShape(0);

  const int32_t batches = in_shape.dim(0);
  assert(in_shape.rank() == 2);
  assert(weights_shape.rank() == 2);
  assert(in_shape.dim(1) == weights_shape.dim(0));
  const int32_t accum_depth = weights_shape.dim(0);
  const int32_t output_depth = weights_shape.dim(1);

  uint8_t *input_data = reinterpret_cast<uint8_t *>(inputv.atOffset(0));
  uint8_t *weights_data = reinterpret_cast<uint8_t *>(weightsv.atOffset(0));
  int32_t *bias_data = reinterpret_cast<int32_t *>(biasv->atOffset(0));

  uint8_t *output_data = reinterpret_cast<uint8_t *>(res.atOffset(0));

  int32_t output_min = std::numeric_limits<uint8_t>::min();
  int32_t output_max = std::numeric_limits<uint8_t>::max();

  for (int32_t b = 0; b < batches; ++b)
  {
    for (int32_t out_c = 0; out_c < output_depth; ++out_c)
    {
      int32_t acc = 0;
      for (int d = 0; d < accum_depth; ++d)
      {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t weights_val = weights_data[d * output_depth + out_c];
        acc += (weights_val + weights_offset) * (input_val + input_offset);
      }
      acc += bias_data[out_c];
      acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc += output_offset;
      acc = std::max(acc, output_min);
      acc = std::min(acc, output_max);
      output_data[out_c + output_depth * b] = static_cast<uint8_t>(acc);
    }
  }
}

void FullyConnected(const mir::TensorVariant &input, const mir::TensorVariant &weights,
                    const mir::ops::FullyConnectedOp &op, mir::TensorVariant &res,
                    const mir::TensorVariant *bias)
{
  dispatch<FullyConnectedImpl>(res.getElementType(), input, weights, op, res, bias);
}
} // namespace mir_interpreter
