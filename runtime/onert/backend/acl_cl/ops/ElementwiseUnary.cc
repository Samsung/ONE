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

      fn = acl_common::generateLayer<arm_compute::CLActivationLayer>(
        input_tensor->handle(), output_tensor->handle(), act_info);
      break;
    }
    case ir::operation::ElementwiseUnary::Type::CAST:
    {
      if (input_tensor->data_type() == output_tensor->data_type())
      {
        fn = acl_common::generateLayer<arm_compute::CLCopy>(input_tensor->handle(),
                                                            output_tensor->handle());
      }
      else if (_ctx.at(input_index).typeInfo().type() == ir::DataType::BOOL8)
      {
        fn = acl_common::generateLayer<arm_compute::CLCastBool>(input_tensor->handle(),
                                                                output_tensor->handle());
      }
      else
      {
        // TODO Support converting float to int32 as round down
        fn = acl_common::generateLayer<arm_compute::CLCast>(
          input_tensor->handle(), output_tensor->handle(), arm_compute::ConvertPolicy::SATURATE);
      }
      break;
    }
    case ir::operation::ElementwiseUnary::Type::DEQUANTIZE:
    {
      fn = acl_common::generateLayer<arm_compute::CLDequantizationLayer>(input_tensor->handle(),
                                                                         output_tensor->handle());
      break;
    }
    case ir::operation::ElementwiseUnary::Type::EXP:
    {
      fn = acl_common::generateLayer<arm_compute::CLExpLayer>(input_tensor->handle(),
                                                              output_tensor->handle());
      break;
    }
    case ir::operation::ElementwiseUnary::Type::FLOOR:
    {
      fn = acl_common::generateLayer<arm_compute::CLFloor>(input_tensor->handle(),
                                                           output_tensor->handle());
      break;
    }
    case ir::operation::ElementwiseUnary::Type::LOGICAL_NOT:
    {
      fn = acl_common::generateLayer<arm_compute::CLBitwiseNot>(input_tensor->handle(),
                                                                output_tensor->handle());
      break;
    }
    case ir::operation::ElementwiseUnary::Type::NEG:
    {
      fn = acl_common::generateLayer<arm_compute::CLNegLayer>(input_tensor->handle(),
                                                              output_tensor->handle());
      break;
    }
    case ir::operation::ElementwiseUnary::Type::RSQRT:
    {
      fn = acl_common::generateLayer<arm_compute::CLRsqrtLayer>(input_tensor->handle(),
                                                                output_tensor->handle());
      break;
    }
    case ir::operation::ElementwiseUnary::Type::SQRT:
    {
      const ::arm_compute::ActivationLayerInfo act_info{
        ::arm_compute::ActivationLayerInfo::ActivationFunction::SQRT};

      fn = acl_common::generateLayer<arm_compute::CLActivationLayer>(
        input_tensor->handle(), output_tensor->handle(), act_info);
      break;
    }
    default:
    {
      throw std::runtime_error("acl_cl KernelGenerator : " + node.name() + "is not supported yet");
      break;
    }
  }

  auto acl_fn = acl_common::asAclFunction(std::move(fn));

  _return_fn = std::move(acl_fn);
}

} // namespace onert::backend::acl_cl
