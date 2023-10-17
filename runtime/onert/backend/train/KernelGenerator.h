/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRAIN_KERNEL_GENERATOR_H__
#define __ONERT_BACKEND_TRAIN_KERNEL_GENERATOR_H__

#include "ExternalContext.h"
#include "backend/basic/TensorRegistry.h"
#include "TensorBuilder.h"
#include "Tensor.h"

#include <backend/train/KernelGeneratorBase.h>
#include <exec/train/IGradientApplier.h>
#include <exec/train/optimizer/Optimizer.h>
#include <ir/Operands.h>
#include <ir/Operations.h>

namespace onert
{
namespace backend
{
namespace train
{

// TODO Unify TensorRegistry
class KernelGenerator : public backend::train::KernelGeneratorBase
{
public:
  KernelGenerator(const ir::train::TrainableGraph &tgraph,
                  const std::shared_ptr<TensorRegistry> &tensor_reg,
                  const std::shared_ptr<ExternalContext> &external_context,
                  const exec::train::optimizer::Optimizer *optimizer);

  std::unique_ptr<exec::train::TrainableFnSequence> generate(ir::OperationIndex op_ind) override;

  void visit(const ir::train::operation::Conv2D &) override;
  void visit(const ir::train::operation::ElementwiseActivation &) override;
  void visit(const ir::train::operation::FullyConnected &) override;
  void visit(const ir::train::operation::Loss &) override;
  void visit(const ir::train::operation::Pool2D &) override;
  void visit(const ir::train::operation::Reshape &node) override;
  void visit(const ir::train::operation::Softmax &node) override;

private:
  ir::Layout _current_layout;
  std::shared_ptr<TensorRegistry> _tensor_reg;
  const std::shared_ptr<ExternalContext> _external_context;
  const exec::train::optimizer::Optimizer *_optimizer;
  std::vector<std::unique_ptr<exec::train::IGradientApplier>> _update_funcs;
};

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_KERNEL_GENERATOR_H__
