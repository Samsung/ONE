/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Builders.h"

#include "kernels/SVDF.h"

namespace luci_interpreter
{

std::unique_ptr<Kernel> build_kernel_CircleSVDF(const luci::CircleNode *circle_node,
                                                KernelBuilderHelper &helper)
{
  const auto *node = loco::must_cast<const luci::CircleSVDF *>(circle_node);
  assert(node->arity() == 5);

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *feature = helper.getInputTensor(node->weight_feature());
  const Tensor *time = helper.getInputTensor(node->weight_time());
  const Tensor *bias = helper.getOptionalInputTensor(node->bias());
  const Tensor *input_activation_state = helper.getInputTensor(node->input_activation_state());
  Tensor *output = helper.getOutputTensor(node);

  auto scratchpad_tensor = std::make_unique<Tensor>(input_activation_state->element_type(),
                                                    Shape({}), AffineQuantization{}, "");
  scratchpad_tensor->set_observable(false);
  scratchpad_tensor->set_data_buffer(nullptr);
  Tensor *tmp = helper.getRuntimeGraph(node->graph())->addTensor(std::move(scratchpad_tensor));

  DataType data_type = input->element_type() == DataType::S8 ? DataType::S32 : DataType::FLOAT32;

  scratchpad_tensor = std::make_unique<Tensor>(data_type, Shape({}), AffineQuantization{}, "");
  scratchpad_tensor->set_observable(false);
  scratchpad_tensor->set_data_buffer(nullptr);
  Tensor *tmp_1 = helper.getRuntimeGraph(node->graph())->addTensor(std::move(scratchpad_tensor));

  if (data_type == DataType::FLOAT32 &&
      (feature->element_type() == DataType::S8 || feature->element_type() == DataType::U8))
  {
    data_type = feature->element_type();
  }

  scratchpad_tensor = std::make_unique<Tensor>(data_type, Shape({}), AffineQuantization{}, "");
  scratchpad_tensor->set_observable(false);
  scratchpad_tensor->set_data_buffer(nullptr);
  Tensor *tmp_2 = helper.getRuntimeGraph(node->graph())->addTensor(std::move(scratchpad_tensor));

  data_type = DataType::FLOAT32;

  scratchpad_tensor = std::make_unique<Tensor>(data_type, Shape({}), AffineQuantization{}, "");
  scratchpad_tensor->set_observable(false);
  scratchpad_tensor->set_data_buffer(nullptr);
  Tensor *tmp_3 = helper.getRuntimeGraph(node->graph())->addTensor(std::move(scratchpad_tensor));

  scratchpad_tensor = std::make_unique<Tensor>(data_type, Shape({}), AffineQuantization{}, "");
  scratchpad_tensor->set_observable(false);
  scratchpad_tensor->set_data_buffer(nullptr);
  Tensor *tmp_4 = helper.getRuntimeGraph(node->graph())->addTensor(std::move(scratchpad_tensor));

  scratchpad_tensor = std::make_unique<Tensor>(data_type, Shape({}), AffineQuantization{}, "");
  scratchpad_tensor->set_observable(false);
  scratchpad_tensor->set_data_buffer(nullptr);
  Tensor *tmp_5 = helper.getRuntimeGraph(node->graph())->addTensor(std::move(scratchpad_tensor));

  scratchpad_tensor = std::make_unique<Tensor>(data_type, Shape({}), AffineQuantization{}, "");
  scratchpad_tensor->set_observable(false);
  scratchpad_tensor->set_data_buffer(nullptr);
  Tensor *tmp_6 = helper.getRuntimeGraph(node->graph())->addTensor(std::move(scratchpad_tensor));

  SVDFParams params{};
  params.activation = node->fusedActivationFunction();
  params.svdf_rank = node->svdf_rank();
  params.asymmetric_quantize_inputs = node->asymmetric_quantize_inputs();

  return std::make_unique<kernels::SVDF>(input, feature, time, bias, input_activation_state, output,
                                         tmp, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, params);
}

} // namespace luci_interpreter
