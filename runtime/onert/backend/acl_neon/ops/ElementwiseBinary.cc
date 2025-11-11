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

void Validator::visit(const ir::operation::ElementwiseBinary &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::ElementwiseBinary &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::ElementwiseBinary::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::ElementwiseBinary::Input::RHS)};

  auto output_tensor = _tensor_reg->getAclTensor(output_index);
  auto lhs_tensor = _tensor_reg->getAclTensor(lhs_index);
  auto rhs_tensor = _tensor_reg->getAclTensor(rhs_index);

  std::unique_ptr<arm_compute::IFunction> fn;
  switch (node.param().op_type)
  {
    case ir::operation::ElementwiseBinary::ElementwiseBinaryType::LOGICAL_AND:
    {
      fn = acl_common::generateLayer<arm_compute::NELogicalAnd>(
        lhs_tensor->handle(), rhs_tensor->handle(), output_tensor->handle());
      break;
    }
    case ir::operation::ElementwiseBinary::ElementwiseBinaryType::LOGICAL_OR:
    {
      fn = acl_common::generateLayer<arm_compute::NELogicalOr>(
        lhs_tensor->handle(), rhs_tensor->handle(), output_tensor->handle());
      break;
    }
    case ir::operation::ElementwiseBinary::ElementwiseBinaryType::MAX:
    {
      fn = acl_common::generateLayer<arm_compute::NEElementwiseMax>(
        lhs_tensor->handle(), rhs_tensor->handle(), output_tensor->handle());
      break;
    }
    case ir::operation::ElementwiseBinary::ElementwiseBinaryType::MIN:
    {
      fn = acl_common::generateLayer<arm_compute::NEElementwiseMin>(
        lhs_tensor->handle(), rhs_tensor->handle(), output_tensor->handle());
      break;
    }
    default:
    {
      std::string err_msg("acl_neon KernelGenerator : " + node.name() +
                          "is not elementwise-binary operations");
      assert(false && err_msg.c_str());
      break;
    }
  }
  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_neon
