/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "BatchMatMul.h"

#include "Convert.h"

namespace circlechef
{

void CircleOpBatchMatMul::filler(const circle::Operator *op, CircleImport *import,
                                 circlechef::ModelRecipe *model_recipe) const
{
  // Nothing to do with filler
}

circlechef::Operation *CircleOpBatchMatMul::build(const circle::Operator *op, CircleImport *import,
                                                  circlechef::ModelRecipe *model_recipe) const
{
  auto op_params = op->builtin_options_as_BatchMatMulOptions();
  assert(op_params != nullptr);

  auto operation = model_recipe->add_operation();

  operation->set_type("BatchMatMul");

  auto op_options = operation->mutable_batch_matmul_options();

  op_options->set_adjoint_lhs(op_params->adjoint_lhs());
  op_options->set_adjoint_rhs(op_params->adjoint_rhs());

  return operation;
}

} // namespace circlechef
