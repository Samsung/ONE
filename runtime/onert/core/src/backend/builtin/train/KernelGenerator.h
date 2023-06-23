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

#ifndef __ONERT_BACKEND_BUTIN_TRAIN_KERNEL_GENERATOR_H__
#define __ONERT_BACKEND_BUTIN_TRAIN_KERNEL_GENERATOR_H__

#include "../ExternalContext.h"
#include "../TensorRegistry.h"
#include "../../../compiler/TensorRegistries.h"

#include <backend/train/KernelGeneratorBase.h>
#include <exec/train/TrainableFnSequence.h>
#include <ir/train/TrainableGraph.h>

namespace onert
{
namespace backend
{
namespace builtin
{
namespace train
{

class KernelGenerator : public backend::train::KernelGeneratorBase
{
public:
  KernelGenerator(const ir::train::TrainableGraph &tgraph,
                  const std::shared_ptr<TensorRegistry> &tensor_reg,
                  const std::shared_ptr<TensorRegistry> &grad_tensor_reg,
                  const std::shared_ptr<ExternalContext> &external_context);

  std::unique_ptr<exec::train::TrainableFnSequence> generate(ir::OperationIndex ind) override;

  void setTensorRegistries(const compiler::TensorRegistries &tensor_registries)
  {
    _tensor_registries = tensor_registries;
  }

private:
  void visit(const ir::train::operation::Permute &) override;

private:
  backend::ITensor *getTensor(const ir::OperandIndex &index);

private:
  std::shared_ptr<TensorRegistry> _tensor_reg;
  std::shared_ptr<TensorRegistry> _grad_tensor_reg;
  compiler::TensorRegistries _tensor_registries;
  const std::shared_ptr<ExternalContext> _external_context;
};

} // namespace train
} // namespace builtin
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BUTIN_TRAIN_KERNEL_GENERATOR_H__
