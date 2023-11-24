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

#ifndef __ONERT_BACKEND_TRAIN_KERNEL_GENERATOR_BASE_H__
#define __ONERT_BACKEND_TRAIN_KERNEL_GENERATOR_BASE_H__

#include <memory>

#include "backend/ITensorRegistry.h"
#include "exec/train/TrainableFnSequence.h"
#include "ir/train/TrainableGraph.h"
#include "ir/OperationVisitor.h"

namespace onert
{
namespace backend
{
namespace train
{

class KernelGeneratorBase : public ir::OperationVisitor
{
public:
  virtual ~KernelGeneratorBase() = default;
  KernelGeneratorBase(const ir::train::TrainableGraph &tgraph) : _tgraph{tgraph} {}

  virtual std::unique_ptr<exec::train::TrainableFnSequence> generate(ir::OperationIndex ind) = 0;

protected:
#define OP(InternalName)                                                                \
  void visit(const ir::operation::InternalName &) override                              \
  {                                                                                     \
    throw std::runtime_error("KernelGenerator: NYI for operation '" #InternalName "'"); \
  }
#include "ir/train/Operations.lst"
#undef OP

protected:
  const ir::train::TrainableGraph &_tgraph;
  std::unique_ptr<exec::train::ITrainableFunction> _return_fn;
};

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_KERNEL_GENERATOR_BASE_H__
