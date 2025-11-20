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

void Validator::visit(const ir::operation::Transpose &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::Transpose &node)
{
  const auto ofm_idx{node.getOutputs().at(0)};
  const auto ifm_idx{node.getInputs().at(ir::operation::Transpose::Input::INPUT)};
  const auto perm_idx{node.getInputs().at(ir::operation::Transpose::Input::PERMUTATION)};

  const auto rank = _ctx.at(ifm_idx).shape().rank();

  auto ofm_tensor = _tensor_reg->getAclTensor(ofm_idx);
  auto ifm_tensor = _tensor_reg->getAclTensor(ifm_idx);

  const auto &perms = _ctx.at(perm_idx);
  std::vector<int32_t> pv;
  if (perms.shape() == ir::Shape{0})
  {
    pv.resize(rank);
    std::iota(pv.begin(), pv.end(), 0);
    std::reverse(pv.begin(), pv.end());
  }
  else
  {
    pv = _ctx.at(perm_idx).asVector<int32_t>();
  }

  std::unique_ptr<arm_compute::IFunction> fn;
  if (rank == 1)
  {
    fn = acl_common::generateLayer<arm_compute::CLCopy>(ifm_tensor->handle(), ofm_tensor->handle());
  }
  else if (rank == 2)
  {
    assert(pv.size() == 2 && pv.at(0) == 1 && pv.at(1) == 0);
    fn = acl_common::generateLayer<arm_compute::CLTranspose>(ifm_tensor->handle(),
                                                             ofm_tensor->handle());
  }
  else
  {
    auto backend_pv = acl_common::getARMComputePermutationVector(rank, pv);

    fn = acl_common::generateLayer<arm_compute::CLPermute>(ifm_tensor->handle(),
                                                           ofm_tensor->handle(), backend_pv);
  }

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_cl
