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

void Validator::visit(const ir::operation::EmbeddingLookup &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::EmbeddingLookup &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lookups_index{node.getInputs().at(ir::operation::EmbeddingLookup::Input::LOOKUPS)};
  const auto values_index{node.getInputs().at(ir::operation::EmbeddingLookup::Input::VALUES)};

  auto output_tensor = _tensor_reg->getAclTensor(output_index);
  auto lookups_tensor = _tensor_reg->getAclTensor(lookups_index);
  auto values_tensor = _tensor_reg->getAclTensor(values_index);

  size_t n = _ctx.at(values_index).shape().rank();
  assert(n == values_tensor->num_dimensions());
  size_t k = _ctx.at(lookups_index).shape().rank();
  assert(k == lookups_tensor->num_dimensions());

  const int axis = ::onert::backend::acl_common::ToARMComputeAxis(n, 0).value();

  // Disable applied dim_correction
  if (n != values_tensor->info()->num_dimensions())
  {
    // This means that high dimension's value is 1 and ifm tensor is applied dim_correction
    acl_common::disableDimCorrection(values_tensor);
  }
  if (k != lookups_tensor->info()->num_dimensions())
  {
    // This means that high dimension's value is 1 and indices tensor is applied dim_correction
    acl_common::disableDimCorrection(lookups_tensor);
  }

  auto fn = acl_common::generateLayer<arm_compute::CLGather>(
    values_tensor->handle(), lookups_tensor->handle(), output_tensor->handle(), axis);

  // Revert disabling applied dim_correction
  if (values_tensor->dimension(0) == 1)
  {
    acl_common::enableDimCorrection(values_tensor);
  }
  if (lookups_tensor->dimension(0) == 1)
  {
    acl_common::enableDimCorrection(lookups_tensor);
  }

  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_cl
