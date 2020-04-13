/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef _MIR_DOT_NODE_BUILDER_H_
#define _MIR_DOT_NODE_BUILDER_H_

#include "DotGraph.h"

#include "mir/Visitor.h"

#include <cstddef>
#include <string>
#include <vector>

namespace mir
{

class DotNodeBuilder : public Visitor
{
public:
  explicit DotNodeBuilder(const Operation &op);

  void visit(ops::AvgPool2DOp &op) override;
  void visit(ops::CappedReluOp &op) override;
  void visit(ops::ConcatOp &op) override;
  void visit(ops::Conv2DOp &op) override;
  void visit(ops::DeConv2DOp &op) override;
  void visit(ops::DepthwiseConv2DOp &op) override;
  void visit(ops::EluOp &op) override;
  void visit(ops::GatherOp &op) override;
  void visit(ops::LeakyReluOp &op) override;
  void visit(ops::MaxPool2DOp &op) override;
  void visit(ops::PadOp &op) override;
  void visit(ops::ReduceMeanOp &op) override;
  void visit(ops::ResizeOp &op) override;
  void visit(ops::SliceOp &op) override;
  void visit(ops::SoftmaxOp &op) override;
  void visit(ops::SqueezeOp &op) override;
  void visit(ops::TransposeOp &op) override;

  void addAttribute(std::string name, std::string val);

  DotNode getDotNode() const { return {_id, getLabel()}; }

private:
  std::string getLabel() const;

  std::size_t _id;
  std::string _type_name;
  std::vector<std::string> _in_shapes;
  std::vector<std::string> _out_shapes;
  std::vector<std::pair<std::string, std::string>> _attributes;
};

} // namespace mir

#endif //_MIR_DOT_NODE_BUILDER_H_
