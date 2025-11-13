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

#include <AclActivationBuilder.h>
#include <AclKernelGen.h>

namespace onert::backend::acl_neon
{

using ActivationBuilder =
  acl_common::AclActivationBuilder<::arm_compute::ITensor, ::arm_compute::NEActivationLayer,
                                   acl_common::AclFunction>;

void Validator::visit(const ir::operation::BinaryArithmetic &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::BinaryArithmetic &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::BinaryArithmetic::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::BinaryArithmetic::Input::RHS)};

  const auto activation = node.param().activation;

  auto ofm_tensor = _tensor_reg->getAclTensor(ofm_index);
  auto lhs_tensor = _tensor_reg->getAclTensor(lhs_index);
  auto rhs_tensor = _tensor_reg->getAclTensor(rhs_index);

  std::unique_ptr<arm_compute::IFunction> fn;
  switch (node.param().arithmetic_type)
  {
    case ir::operation::BinaryArithmetic::ArithmeticType::ADD:
    {
      arm_compute::NEArithmeticAddition::validate(lhs_tensor->info(), rhs_tensor->info(),
                                                  ofm_tensor->info(),
                                                  arm_compute::ConvertPolicy::SATURATE)
        .throw_if_error();
      fn = acl_common::generateLayer<arm_compute::NEArithmeticAddition>(
        lhs_tensor->handle(), rhs_tensor->handle(), ofm_tensor->handle(),
        arm_compute::ConvertPolicy::SATURATE);
      break;
    }
    case ir::operation::BinaryArithmetic::ArithmeticType::SUB:
    {
      arm_compute::NEArithmeticSubtraction::validate(lhs_tensor->info(), rhs_tensor->info(),
                                                     ofm_tensor->info(),
                                                     arm_compute::ConvertPolicy::SATURATE)
        .throw_if_error();
      fn = acl_common::generateLayer<arm_compute::NEArithmeticSubtraction>(
        lhs_tensor->handle(), rhs_tensor->handle(), ofm_tensor->handle(),
        arm_compute::ConvertPolicy::SATURATE);
      break;
    }
    case ir::operation::BinaryArithmetic::ArithmeticType::MUL:
    {
      arm_compute::NEPixelWiseMultiplication::validate(
        lhs_tensor->info(), rhs_tensor->info(), ofm_tensor->info(), 1.0,
        arm_compute::ConvertPolicy::SATURATE, arm_compute::RoundingPolicy::TO_ZERO)
        .throw_if_error();
      // RoundingPolicy for scale:1.0 is only allowed RoundingPolicy::TO_ZERO
      fn = acl_common::generateLayer<arm_compute::NEPixelWiseMultiplication>(
        lhs_tensor->handle(), rhs_tensor->handle(), ofm_tensor->handle(), 1.0, // scale
        arm_compute::ConvertPolicy::SATURATE, arm_compute::RoundingPolicy::TO_ZERO);
      break;
    }
    case ir::operation::BinaryArithmetic::ArithmeticType::DIV:
    {
      arm_compute::NEElementwiseDivision::validate(lhs_tensor->info(), rhs_tensor->info(),
                                                   ofm_tensor->info())
        .throw_if_error();
      fn = acl_common::generateLayer<arm_compute::NEElementwiseDivision>(
        lhs_tensor->handle(), rhs_tensor->handle(), ofm_tensor->handle());
      break;
    }
    default:
      assert(false && "The BinaryArithmetic operation supports only binary arithmetic operations");
      break;
  }
  _return_fn = std::make_unique<exec::FunctionSequence>(
    acl_common::asAclFunction(std::move(fn)),
    ActivationBuilder::generate(activation, ofm_tensor->handle()));
}

} // namespace onert::backend::acl_neon
