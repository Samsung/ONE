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

std::unique_ptr<Kernel>
build_kernel_CircleSVDF(std::vector<std::pair<const Tensor *, int32_t>> &inputs,
                        std::vector<std::pair<Tensor *, int32_t>> &outputs, const uint32_t op_index,
                        KernelBuilder &builder)
{
  assert(inputs.size() == 5);

  const Tensor *input = inputs.at(0).first;
  const Tensor *feature = inputs.at(1).first;
  const Tensor *time = inputs.at(2).first;
  const Tensor *bias = inputs.at(3).first;
  const Tensor *input_activation_state = inputs.at(4).first;
  Tensor *output = outputs.at(0).first;

  auto scratchpad_tensor =
    std::make_unique<Tensor>(input_activation_state->element_type(), Shape({}), nullptr);
  scratchpad_tensor->set_data_buffer(nullptr);
  Tensor *tmp = builder.get_runtime_graph()->addTensor(std::move(scratchpad_tensor));

  DataType data_type = input->element_type() == DataType::S8 ? DataType::S32 : DataType::FLOAT32;

  scratchpad_tensor = std::make_unique<Tensor>(data_type, Shape({}), nullptr);
  scratchpad_tensor->set_data_buffer(nullptr);
  Tensor *tmp_1 = builder.get_runtime_graph()->addTensor(std::move(scratchpad_tensor));

  if (data_type == DataType::FLOAT32 &&
      (feature->element_type() == DataType::S8 || feature->element_type() == DataType::U8))
  {
    data_type = feature->element_type();
  }

  scratchpad_tensor = std::make_unique<Tensor>(data_type, Shape({}), nullptr);
  scratchpad_tensor->set_data_buffer(nullptr);
  Tensor *tmp_2 = builder.get_runtime_graph()->addTensor(std::move(scratchpad_tensor));

  data_type = DataType::FLOAT32;

  scratchpad_tensor = std::make_unique<Tensor>(data_type, Shape({}), nullptr);
  scratchpad_tensor->set_data_buffer(nullptr);
  Tensor *tmp_3 = builder.get_runtime_graph()->addTensor(std::move(scratchpad_tensor));

  scratchpad_tensor = std::make_unique<Tensor>(data_type, Shape({}), nullptr);
  scratchpad_tensor->set_data_buffer(nullptr);
  Tensor *tmp_4 = builder.get_runtime_graph()->addTensor(std::move(scratchpad_tensor));

  scratchpad_tensor = std::make_unique<Tensor>(data_type, Shape({}), nullptr);
  scratchpad_tensor->set_data_buffer(nullptr);
  Tensor *tmp_5 = builder.get_runtime_graph()->addTensor(std::move(scratchpad_tensor));

  scratchpad_tensor = std::make_unique<Tensor>(data_type, Shape({}), nullptr);
  scratchpad_tensor->set_data_buffer(nullptr);
  Tensor *tmp_6 = builder.get_runtime_graph()->addTensor(std::move(scratchpad_tensor));

  circle::OperatorT oper_t;
  builder.get_circle_reader()->operators()[op_index]->UnPackTo(&oper_t);
  const auto *options = oper_t.builtin_options.AsSVDFOptions();

  SVDFParams params{};
  params.activation = luci_actfunc(options->fused_activation_function);
  params.svdf_rank = options->rank;
  params.asymmetric_quantize_inputs = options->asymmetric_quantize_inputs;

  return std::make_unique<kernels::SVDF>(input, feature, time, bias, input_activation_state, output,
                                         tmp, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6, params);
}

} // namespace luci_interpreter
