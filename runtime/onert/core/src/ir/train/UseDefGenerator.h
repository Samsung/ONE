/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_TRAIN_USEDEFINITIALIZER_H__
#define __ONERT_IR_TRAIN_USEDEFINITIALIZER_H__

#include "ir/train/TrainableOperationVisitor.h"

#include "ir/train/UseDefChains.h"
#include "ir/train/Operations.Include.h"

namespace onert::ir::train
{
class TrainableGraph;
} // namespace onert::ir::train

namespace onert::ir::train
{

struct UseDefGeneratorBase : public TrainableOperationVisitor
{
  virtual ~UseDefGeneratorBase() = default;

protected:
#define OP(InternalName)                                                                \
  virtual void visit(const operation::InternalName &) override                          \
  {                                                                                     \
    throw std::runtime_error("UseDefGenerator: NYI for operation '" #InternalName "'"); \
  }
#include "ir/train/Operations.lst"
#undef OP
};

class UseDefGenerator : public UseDefGeneratorBase
{
public:
  UseDefGenerator(void) = delete;
  UseDefGenerator(const TrainableGraph &tgraph);

public:
  UseDefChains operator()();

public:
  void visit(const train::operation::BinaryArithmetic &node) override;
  void visit(const train::operation::Conv2D &node) override;
  void visit(const train::operation::DepthwiseConv2D &node) override;
  void visit(const train::operation::ElementwiseActivation &node) override;
  void visit(const train::operation::FullyConnected &node) override;
  void visit(const train::operation::Loss &node) override;
  void visit(const train::operation::Pad &node) override;
  void visit(const train::operation::Pool2D &node) override;
  void visit(const train::operation::Reduce &node) override;
  void visit(const train::operation::Reshape &node) override;
  void visit(const train::operation::Softmax &node) override;

private:
  void insertUse(const TrainingOperandIndex &operand_index, const TrainingOperationIndex &op_index);
  void insertDef(const TrainingOperandIndex &operand_index, const TrainingOperationIndex &op_index);
  void insertBackPropDef(const TrainingOperandIndex &operand_index,
                         const TrainingOperationIndex &op_index);
  void initForForwardingNodes();
  void initForBackwardingNodes();

private:
  const TrainableGraph &_tgraph;
  std::unordered_map<const ITrainableOperation *, OperationIndex> _node_to_idx;
  UseDefChains _training_usedefs;
};

} // namespace onert::ir::train

#endif // __ONERT_IR_TRAIN_USEDEFINITIALIZER_H__
