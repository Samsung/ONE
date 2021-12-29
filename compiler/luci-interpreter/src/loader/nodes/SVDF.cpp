/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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
  const auto *node = dynamic_cast<const luci::CircleSVDF *>(circle_node);
  if (node == nullptr)
    throw std::runtime_error("wrong builder for operation");

  const Tensor *input = helper.getInputTensor(node->input());
  const Tensor *feature = helper.getInputTensor(node->weight_feature());
  const Tensor *time = helper.getInputTensor(node->weight_time());
  const Tensor *bias = helper.getInputTensor(node->bias());
  const Tensor *input_activation_state = helper.getInputTensor(node->input_activation_state());
  Tensor *output = helper.getOutputTensor(node);

  auto scratchpad_activation_state = std::make_unique<Tensor>(
    input_activation_state->element_type(), Shape({}), AffineQuantization{}, "");
  scratchpad_activation_state->set_observable(false);
  scratchpad_activation_state->set_data_buffer(nullptr);

  // It is unknown what data will be stored in scratchpad tensor,
  // using UINT8 as a most general option
  auto scratchpad_1 = std::make_unique<Tensor>(DataType::U8, Shape({}), AffineQuantization{}, "");
  scratchpad_1->set_observable(false);
  scratchpad_1->set_data_buffer(nullptr);

  // It is unknown what data will be stored in scratchpad tensor,
  // using UINT8 as a most general option
  auto scratchpad_2 = std::make_unique<Tensor>(DataType::U8, Shape({}), AffineQuantization{}, "");
  scratchpad_2->set_observable(false);
  scratchpad_2->set_data_buffer(nullptr);

  // It is unknown what data will be stored in scratchpad tensor,
  // using UINT8 as a most general option
  auto scratchpad_3 = std::make_unique<Tensor>(DataType::U8, Shape({}), AffineQuantization{}, "");
  scratchpad_3->set_observable(false);
  scratchpad_3->set_data_buffer(nullptr);

  // It is unknown what data will be stored in scratchpad tensor,
  // using UINT8 as a most general option
  auto scratchpad_4 = std::make_unique<Tensor>(DataType::U8, Shape({}), AffineQuantization{}, "");
  scratchpad_4->set_observable(false);
  scratchpad_4->set_data_buffer(nullptr);

  // It is unknown what data will be stored in scratchpad tensor,
  // using UINT8 as a most general option
  auto scratchpad_5 = std::make_unique<Tensor>(DataType::U8, Shape({}), AffineQuantization{}, "");
  scratchpad_5->set_observable(false);
  scratchpad_5->set_data_buffer(nullptr);

  // It is unknown what data will be stored in scratchpad tensor,
  // using UINT8 as a most general option
  auto scratchpad_6 = std::make_unique<Tensor>(DataType::U8, Shape({}), AffineQuantization{}, "");
  scratchpad_6->set_observable(false);
  scratchpad_6->set_data_buffer(nullptr);

  Tensor *tmp =
    helper.getRuntimeGraph(node->graph())->addTensor(std::move(scratchpad_activation_state));
  Tensor *tmp_1 = helper.getRuntimeGraph(node->graph())->addTensor(std::move(scratchpad_1));
  Tensor *tmp_2 = helper.getRuntimeGraph(node->graph())->addTensor(std::move(scratchpad_2));
  Tensor *tmp_3 = helper.getRuntimeGraph(node->graph())->addTensor(std::move(scratchpad_3));
  Tensor *tmp_4 = helper.getRuntimeGraph(node->graph())->addTensor(std::move(scratchpad_4));
  Tensor *tmp_5 = helper.getRuntimeGraph(node->graph())->addTensor(std::move(scratchpad_5));
  Tensor *tmp_6 = helper.getRuntimeGraph(node->graph())->addTensor(std::move(scratchpad_6));

  SVDFParams params{};
  params.activation = node->fusedActivationFunction();
  params.svdf_rank = node->svdf_rank();
  params.asymmetric_quantize_inputs = node->asymmetric_quantize_inputs();

  return std::make_unique<kernels::SVDF>(input, feature, time, bias, input_activation_state, output,
                                         tmp, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, params);
}

} // namespace luci_interpreter
