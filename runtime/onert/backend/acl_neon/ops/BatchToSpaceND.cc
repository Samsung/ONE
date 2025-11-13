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

void Validator::visit(const ir::operation::BatchToSpaceND &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::BatchToSpaceND &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::BatchToSpaceND::Input::INPUT)};
  const auto block_size_index{
    node.getInputs().at(ir::operation::BatchToSpaceND::Input::BLOCK_SIZE)};

  const auto NNApiInputs = 2;
  if (node.getInputs().size() != NNApiInputs)
  {
    const auto crops_index{node.getInputs().at(ir::operation::BatchToSpaceND::Input::CROPS_DATA)};
    if (!_ctx.at(crops_index).isConstant())
    {
      throw std::runtime_error("Non-constant crops NYI for acl_neon backend BatchToSpaceND");
    }

    auto crops = _ctx.at(crops_index).asVector<int32_t>();
    for (auto &&crop : crops)
    {
      if (crop != 0)
      {
        throw std::runtime_error("Non-zero crops NYI for acl_neon backend BatchToSpaceND");
      }
    }
  }

  auto ofm_tensor = _tensor_reg->getAclTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getAclTensor(ifm_index);

  if (!_ctx.at(block_size_index).data())
    throw std::runtime_error("ACL NEON does not support dynamic block size for BatchToSpaceND");

  auto block = _ctx.at(block_size_index).asVector<int32_t>();
  int32_t height = block[0];
  int32_t width = block[1];

  auto fn = acl_common::generateLayer<arm_compute::NEBatchToSpaceLayer>(
    ifm_tensor->handle(), width, height, ofm_tensor->handle());

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_neon
