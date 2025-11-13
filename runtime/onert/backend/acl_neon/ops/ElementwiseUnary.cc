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

void Validator::visit(const ir::operation::ElementwiseUnary &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::ElementwiseUnary &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::ElementwiseUnary::Input::INPUT)};

  auto output_tensor = _tensor_reg->getAclTensor(output_index);
  auto input_tensor = _tensor_reg->getAclTensor(input_index);

  std::unique_ptr<arm_compute::IFunction> fn;
  switch (node.param().op_type)
  {
    case ir::operation::ElementwiseUnary::Type::ABS:
    {
      const ::arm_compute::ActivationLayerInfo act_info{
        ::arm_compute::ActivationLayerInfo::ActivationFunction::ABS};

      fn = acl_common::generateLayer<arm_compute::NEActivationLayer>(
        input_tensor->handle(), output_tensor->handle(), act_info);
      break;
    }
    case ir::operation::ElementwiseUnary::Type::CAST:
    {
      if (input_tensor->data_type() == output_tensor->data_type())
      {
        fn = acl_common::generateLayer<arm_compute::NECopy>(input_tensor->handle(),
                                                            output_tensor->handle());
      }
      else if (_ctx.at(input_index).typeInfo().type() == ir::DataType::BOOL8)
      {
        fn = acl_common::generateLayer<arm_compute::NECastBool>(input_tensor->handle(),
                                                                output_tensor->handle());
      }
      else
      {
        fn = acl_common::generateLayer<arm_compute::NECast>(
          input_tensor->handle(), output_tensor->handle(), arm_compute::ConvertPolicy::SATURATE);
      }
      break;
    }
    case ir::operation::ElementwiseUnary::Type::DEQUANTIZE:
    {
      fn = acl_common::generateLayer<arm_compute::NEDequantizationLayer>(input_tensor->handle(),
                                                                         output_tensor->handle());
      break;
    }
    case ir::operation::ElementwiseUnary::Type::EXP:
    {
      fn = acl_common::generateLayer<arm_compute::NEExpLayer>(input_tensor->handle(),
                                                              output_tensor->handle());
      break;
    }
    case ir::operation::ElementwiseUnary::Type::FLOOR:
    {
      fn = acl_common::generateLayer<arm_compute::NEFloor>(input_tensor->handle(),
                                                           output_tensor->handle());
      break;
    }
    case ir::operation::ElementwiseUnary::Type::LOGICAL_NOT:
    {
      fn = acl_common::generateLayer<arm_compute::NEBitwiseNot>(input_tensor->handle(),
                                                                output_tensor->handle());
      break;
    }
    case ir::operation::ElementwiseUnary::Type::NEG:
    {
      fn = acl_common::generateLayer<arm_compute::NENegLayer>(input_tensor->handle(),
                                                              output_tensor->handle());
      break;
    }
    case ir::operation::ElementwiseUnary::Type::RSQRT:
    {
      fn = acl_common::generateLayer<arm_compute::NERsqrtLayer>(input_tensor->handle(),
                                                                output_tensor->handle());
      break;
    }
    case ir::operation::ElementwiseUnary::Type::SQRT:
    {
      const ::arm_compute::ActivationLayerInfo act_info{
        ::arm_compute::ActivationLayerInfo::ActivationFunction::SQRT};

      fn = acl_common::generateLayer<arm_compute::NEActivationLayer>(
        input_tensor->handle(), output_tensor->handle(), act_info);
      break;
    }
    default:
    {
      throw std::runtime_error("acl_neon KernelGenerator : " + node.name() +
                               "is not supported yet");
      break;
    }
  }
  _return_fn = acl_common::asAclFunction(std::move(fn));
}

} // namespace onert::backend::acl_neon
