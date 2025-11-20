/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../KernelGenerator.h"
#include "../Validator.h"

#include <AclKernelGen.h>

namespace onert::backend::acl_neon
{

void Validator::visit(const ir::operation::RNN &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::RNN &node)
{
  const auto output_index{node.getOutputs().at(ir::operation::RNN::Output::OUTPUT)};
  const auto hidden_state_out_index{
    node.getOutputs().at(ir::operation::RNN::Output::HIDDEN_STATE_OUT)};

  const auto input_index{node.getInputs().at(ir::operation::RNN::Input::INPUT)};
  const auto weights_index{node.getInputs().at(ir::operation::RNN::Input::WEIGHTS)};
  const auto recurrent_weights_index{
    node.getInputs().at(ir::operation::RNN::Input::RECURRENT_WEIGHTS)};
  const auto bias_index{node.getInputs().at(ir::operation::RNN::Input::BIAS)};
  const auto hidden_state_in_index{node.getInputs().at(ir::operation::RNN::Input::HIDDEN_STATE_IN)};

  const auto activation = node.param().activation;

  auto output_tensor = _tensor_reg->getAclTensor(output_index);
  auto hidden_state_out_tensor = _tensor_reg->getAclTensor(hidden_state_out_index);

  auto input_tensor = _tensor_reg->getAclTensor(input_index);
  auto weights_tensor = _tensor_reg->getAclTensor(weights_index);
  auto recurrent_weights_tensor = _tensor_reg->getAclTensor(recurrent_weights_index);
  auto bias_tensor = _tensor_reg->getAclTensor(bias_index);
  auto hidden_state_in_tensor = _tensor_reg->getAclTensor(hidden_state_in_index);
  auto act_info = ::onert::backend::acl_common::asActivationLayerInfo(activation);

  auto copy_layer = acl_common::generateLayer<arm_compute::NECopy>(
    hidden_state_in_tensor->handle(), hidden_state_out_tensor->handle());
  _return_fn = acl_common::asAclFunction(std::move(copy_layer));

  auto fn = acl_common::generateLayer<arm_compute::NERNNLayer>(
    _tensor_builder->acl_tensor_manager()->internal_buffer_manager(), input_tensor->handle(),
    weights_tensor->handle(), recurrent_weights_tensor->handle(), bias_tensor->handle(),
    hidden_state_out_tensor->handle(), output_tensor->handle(), act_info);
  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_neon
