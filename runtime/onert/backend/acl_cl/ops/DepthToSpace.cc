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

namespace onert::backend::acl_cl
{

void Validator::visit(const ir::operation::DepthToSpace &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::DepthToSpace &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::DepthToSpace::Input::INPUT)};

  auto block_size = node.param().block_size;
  assert(block_size > 0);

  auto output_tensor = _tensor_reg->getAclTensor(output_index);
  auto input_tensor = _tensor_reg->getAclTensor(input_index);

  auto fn = acl_common::generateLayer<arm_compute::CLDepthToSpaceLayer>(
    input_tensor->handle(), output_tensor->handle(), block_size);

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_cl
