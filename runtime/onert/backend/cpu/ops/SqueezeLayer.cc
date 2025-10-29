/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in riting, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Empty file for Squeeze operation
// Sqeeze operation is implemented as ReshapeLayer

#include "ReshapeLayer.h"
#include "../KernelGenerator.h"
#include "../Validator.h"

namespace onert::backend::cpu
{

void Validator::visit(const ir::operation::Squeeze &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::Squeeze &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Squeeze::Input::INPUT)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  // Squeeze can share same kernel with reshape
  auto fn = std::make_unique<ops::ReshapeLayer>();

  fn->configure(input_tensor, nullptr, output_tensor);

  _return_fn = std::move(fn);
}

} // namespace onert::backend::cpu
