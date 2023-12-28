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

#ifndef __ONERT_COMPILER_TRAIN_STATIC_BACK_PROP_SHAPE_INFERER_H__
#define __ONERT_COMPILER_TRAIN_STATIC_BACK_PROP_SHAPE_INFERER_H__

#include "ir/train/TrainableOperationVisitor.h"

#include "compiler/train/LoweredTrainableGraph.h"
#include "ir/Index.h"

#include <memory>
#include <unordered_map>

namespace onert
{
namespace compiler
{
namespace train
{

/**
 * @brief Class to infer shape before running kernels. It does the following:
 *        - re-calculate and set output shape at compile time (before running kernels)
 *        - if calculation cannot be done at compile time, mark the outputs to be dynamic, meaning
 *          shapes of outputs will be calculated during running kernels
 */
class StaticBackPropShapeInferer : public ir::train::TrainableOperationVisitor
{
public:
  StaticBackPropShapeInferer(compiler::train::LoweredTrainableGraph *lowered_subg)
    : _lowered_subg{lowered_subg}
  {
  }

  /**
   * @brief Infer shape of operands belonging to ops and set the output shape.
   *        If output shape cannot be known without running op, mark it so that it can be allocated
   *        when running kernel.
   */
  void infer(void);

  void dump();

private:
  bool checkDynamicInput(const ir::IOperation &op);
  void checkOutput(const ir::IOperation &op);
  void setShape(const ir::OperandIndex &index, const ir::Shape &shape);

private:
  void visit(const ir::train::operation::Conv2D &op) override;
  void visit(const ir::train::operation::ElementwiseActivation &op) override;
  void visit(const ir::train::operation::Loss &op) override;
  void visit(const ir::train::operation::Permute &op) override;
  void visit(const ir::train::operation::Pool2D &op) override;
  void visit(const ir::train::operation::Reshape &op) override;
  void visit(const ir::train::operation::Softmax &op) override;

private:
  compiler::train::LoweredTrainableGraph *_lowered_subg;
};

} // namespace train
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_TRAIN_STATIC_BACK_PROP_SHAPE_INFERER_H__
