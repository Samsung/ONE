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

#include "kernels/Prelu.h"

#include "kernels/BinaryOpCommon.h"
#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

Prelu::Prelu(const Tensor *input, const Tensor *alpha, Tensor *output)
  : Kernel({input, alpha}, {output})
{
}

Prelu::~Prelu()
{
  // Destructor declared to delete vector of alpha quantized data properly
}

void Prelu::configure()
{
  LUCI_INTERPRETER_CHECK(input()->element_type() == output()->element_type());
  LUCI_INTERPRETER_CHECK(alpha()->element_type() == output()->element_type());
  LUCI_INTERPRETER_CHECK(input()->scales().size() <= 1);
  LUCI_INTERPRETER_CHECK(output()->scales().size() <= 1);

  if (input()->element_type() == DataType::U8)
  {
    LUCI_INTERPRETER_CHECK(alpha()->scales().size() <= 1);
    _alpha_multipliers.resize(1);
    double alpha_multiplier = input()->scale() * alpha()->scale() / output()->scale();
    quantizeMultiplier(alpha_multiplier, &_alpha_multipliers[0].multiplier,
                       &_alpha_multipliers[0].shift);
    double identity_multiplier = input()->scale() / output()->scale();
    quantizeMultiplier(identity_multiplier, &_output_multiplier_identity, &_output_shift_identity);
  }
  else if (input()->element_type() == DataType::S16)
  {
    LUCI_INTERPRETER_CHECK(input()->zero_point() == 0 && output()->zero_point() == 0);
    for (size_t channel = 0; channel < alpha()->zero_points().size(); ++channel)
    {
      LUCI_INTERPRETER_CHECK(alpha()->zero_points()[channel] == 0);
    }

    std::vector<double> real_multipliers =
        getQuantizedConvolutionMultiplers(input()->scale(), alpha()->scales(), output()->scale());

    _alpha_multipliers = quantizeMultipliers(real_multipliers);

    double identity_multiplier = input()->scale() / output()->scale();
    quantizeMultiplier(identity_multiplier, &_output_multiplier_identity, &_output_shift_identity);
  }
  output()->resize(calculateShapeForBroadcast(input()->shape(), alpha()->shape()));
}

void Prelu::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    case DataType::U8:
      evalQuantized();
      break;
    case DataType::S16:
      evalQuantizedS16();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void Prelu::evalFloat() const
{
  const auto input_data = getTensorData<float>(input());
  const auto alpha_data = getTensorData<float>(alpha());
  const auto size = getTensorShape(input()).FlatSize();
  auto output_data = getTensorData<float>(output());

  auto PreluFunc = [](float input, float alpha) { return input >= 0.0 ? input : input * alpha; };

  if (input()->shape() != alpha()->shape())
  {
    tflite::reference_ops::BroadcastBinaryFunction4DSlow<float, float, float>(
      getTensorShape(input()), getTensorData<float>(input()), getTensorShape(alpha()),
      getTensorData<float>(alpha()), getTensorShape(output()), getTensorData<float>(output()),
      PreluFunc);
  }
  else
  {
    for (auto i = decltype(size){0}; i < size; ++i)
    {
      if (input_data[i] >= 0)
        output_data[i] = input_data[i];
      else
        output_data[i] = input_data[i] * alpha_data[i];
    }
  }
}

void Prelu::evalQuantized() const
{
  tflite::PreluParams op_params{};

  op_params.input_offset = -input()->zero_point(); // Note the '-'.
  op_params.alpha_offset = -alpha()->zero_point(); // Note the '-'.
  op_params.output_offset = output()->zero_point();
  op_params.output_shift_1 = _output_shift_identity;
  op_params.output_multiplier_1 = _output_multiplier_identity;
  op_params.output_shift_2 = _alpha_multipliers[0].shift;
  op_params.output_multiplier_2 = _alpha_multipliers[0].multiplier;

  if (input()->shape() != alpha()->shape())
  {
    tflite::reference_ops::BroadcastPrelu4DSlow(
      op_params, getTensorShape(input()), getTensorData<uint8_t>(input()), getTensorShape(alpha()),
      getTensorData<uint8_t>(alpha()), getTensorShape(output()), getTensorData<uint8_t>(output()));
  }
  else
  {
    tflite::reference_ops::Prelu<uint8_t>(
      op_params, getTensorShape(input()), getTensorData<uint8_t>(input()), getTensorShape(alpha()),
      getTensorData<uint8_t>(alpha()), getTensorShape(output()), getTensorData<uint8_t>(output()));
  }
}

static inline int16_t evalElemS16Prelu(int16_t input_val, int16_t alpha_val,
                                   const ChannelQuantMultipliers &identity_mult,
                                   const ChannelQuantMultipliers &alpha_mult)
{
  constexpr int32_t quantized_min = std::numeric_limits<int16_t>::min();
  constexpr int32_t quantized_max = std::numeric_limits<int16_t>::max();

  const int32_t output_val =
    input_val >= 0 ? tflite::MultiplyByQuantizedMultiplier(input_val, identity_mult.multiplier,
                                                           identity_mult.shift)
                   : tflite::MultiplyByQuantizedMultiplier(input_val * alpha_val,
                                                           alpha_mult.multiplier,
                                                           alpha_mult.shift);
  const int32_t clamped_output = std::min(quantized_max, std::max(quantized_min, output_val));
  return clamped_output;
}

void Prelu::evalQuantizedS16() const
{
  bool channel_wise = alpha()->scales().size() != 1;
  if (!channel_wise)
  {
    auto fn = [this](int16_t input_val, int16_t alpha_val) {
      const ChannelQuantMultipliers pos_mult{_output_shift_identity, _output_multiplier_identity};
      const ChannelQuantMultipliers &neg_mult = _alpha_multipliers[0];
      return evalElemS16Prelu(input_val, alpha_val, pos_mult, neg_mult);
    };

    BinaryOpBroadcastSlow(getTensorShape(input()), getTensorData<int16_t>(input()),
                          getTensorShape(alpha()), getTensorData<int16_t>(alpha()),
                          getTensorShape(output()), getTensorData<int16_t>(output()), fn);
  }
  else
  {
    tflite::RuntimeShape input_shape = getTensorShape(input());
    tflite::RuntimeShape alpha_shape = getTensorShape(alpha());
    tflite::RuntimeShape output_shape = getTensorShape(output());
    const int16_t *input_data = input()->data<int16_t>();
    const int16_t *alpha_data = alpha()->data<int16_t>();
    int16_t *output_data = output()->data<int16_t>();

    constexpr int N = 5;

    const ChannelQuantMultipliers pos_mult{_output_shift_identity, _output_multiplier_identity};

    if (input_shape == alpha_shape)
    {
      const int flat_size = tflite::MatchingElementsSize(
          input_shape, alpha_shape, output_shape);
      int quantized_dim = alpha()->quantized_dimension();

      size_t outer_dims_size = 1;
      for (int i = 0; i < quantized_dim; ++i)
        outer_dims_size *= input_shape.Dims(i);
      size_t quant_dim_size = alpha_shape.Dims(alpha()->quantized_dimension());
      size_t inner_dims_size = flat_size / outer_dims_size / quant_dim_size;

      for (int outer_dims = 0; outer_dims < outer_dims_size; ++outer_dims)
        for (int quant_channel = 0; quant_channel < quant_dim_size; ++quant_channel)
        {
          const ChannelQuantMultipliers &neg_mult = _alpha_multipliers[quant_channel];
          for (int inner_dims = 0; inner_dims < inner_dims_size; ++inner_dims)
          {
            size_t offset =
                inner_dims + (quant_channel + outer_dims * quant_dim_size) * inner_dims_size;
            output_data[offset] =
                evalElemS16Prelu(input_data[offset], alpha_data[offset], pos_mult, neg_mult);
          }
        }
    }
    else
    {
      assert(input_shape.DimensionsCount() <= N);
      assert(alpha_shape.DimensionsCount() <= N);
      assert(output_shape.DimensionsCount() <= N);

      tflite::NdArrayDesc<N> desc1{};
      tflite::NdArrayDesc<N> desc2{};
      tflite::NdArrayDesc<N> output_desc{};
      tflite::NdArrayDescsForElementwiseBroadcast(input_shape, alpha_shape,
                                                  &desc1, &desc2);
      tflite::CopyDimsToDesc(tflite::RuntimeShape::ExtendedShape(N, output_shape),
                             &output_desc);

      const int quantized_dim = alpha()->quantized_dimension() + N - alpha()->shape().num_dims();

      auto fn = [&](int indexes[N]) {
        const ChannelQuantMultipliers &neg_mult = _alpha_multipliers[indexes[quantized_dim]];
        output_data[SubscriptToIndex(output_desc, indexes)] =
            evalElemS16Prelu(input_data[SubscriptToIndex(desc1, indexes)],
               alpha_data[SubscriptToIndex(desc2, indexes)],
               pos_mult,
               neg_mult);
      };
      tflite::NDOpsHelper<N>(output_desc, fn);
    }
  }

}

} // namespace kernels
} // namespace luci_interpreter
