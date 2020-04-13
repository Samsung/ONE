/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "mir/Graph.h"
#include "mir/Visitor.h"
#include "loco.h"

namespace mir2loco
{

class Transformer final : public mir::Visitor
{
public:
  Transformer() = default;
  ~Transformer() = default;

  void visit(mir::ops::AddOp &op) override;
  void visit(mir::ops::AvgPool2DOp &op) override;
  void visit(mir::ops::ConcatOp &op) override;
  void visit(mir::ops::ConstantOp &op) override;
  void visit(mir::ops::Conv2DOp &op) override;
  void visit(mir::ops::DeConv2DOp &op) override;
  void visit(mir::ops::DepthwiseConv2DOp &op) override;
  void visit(mir::ops::DivOp &op) override;
  void visit(mir::ops::FullyConnectedOp &op) override;
  void visit(mir::ops::InputOp &op) override;
  void visit(mir::ops::MaxPool2DOp &op) override;
  void visit(mir::ops::MulOp &op) override;
  void visit(mir::ops::OutputOp &op) override;
  void visit(mir::ops::ReluOp &op) override;
  void visit(mir::ops::ReshapeOp &op) override;
  void visit(mir::ops::SoftmaxOp &op) override;
  void visit(mir::ops::SubOp &op) override;
  void visit(mir::ops::TransposeOp &op) override;

  void visit_fallback(mir::Operation &op) override;

  std::unique_ptr<loco::Graph> transform(mir::Graph *mir_graph);

private:
  std::unique_ptr<loco::Graph> _loco_graph;
  std::unordered_map<mir::Operation::Output *, loco::Node *> _mir2loco_map;
};

} // namespace mir2loco
