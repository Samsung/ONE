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
#include "exec/NopFunction.h"

namespace onert::backend::acl_cl
{

void Validator::visit(const ir::operation::Concat &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::Concat &node)
{
  const auto ofm_index{node.getOutputs().at(0)};

  std::vector<ir::OperandIndex> input_indexes;

  for (const auto &input : node.getInputs())
    input_indexes.emplace_back(input);

  const auto axis = node.param().axis;

  // Concat elimination check
  bool eliminated = _tensor_builder->areSubTensorsOf(ofm_index, node.getInputs());
  if (eliminated)
  {
    // If concat eliminated, return a NOP IFunction
    VERBOSE(acl_cl_KernelGenerator_Concat) << "Concat eliminated" << std::endl;
    _return_fn = std::make_unique<exec::NopFunction>();
    return;
  }

  auto output_tensor = _tensor_reg->getAclTensor(ofm_index);
  std::vector<const ::arm_compute::ICLTensor *> input_tensors;
  for (const auto &ifm_ind : input_indexes)
    input_tensors.emplace_back(_tensor_reg->getAclTensor(ifm_ind)->handle());

  std::unique_ptr<::arm_compute::IFunction> fn;
  if (input_indexes.size() < 2)
  {
    ::arm_compute::ICLTensor *input_tesor =
      _tensor_reg->getAclTensor(input_indexes.at(0))->handle();

    fn = acl_common::generateLayer<arm_compute::CLCopy>(input_tesor, output_tensor->handle());
  }
  else
  {
    const auto rank = _ctx.at(ofm_index).shape().rank();
    const auto fixed_axis = acl_common::ToARMComputeAxis(rank, axis).value();
    fn = acl_common::generateLayer<::arm_compute::CLConcatenateLayer>(
      input_tensors, output_tensor->handle(), fixed_axis);
  }

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_cl
