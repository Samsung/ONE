/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NEURUN_IR_OP_SEQUENCE_H__
#define __NEURUN_IR_OP_SEQUENCE_H__

#include <vector>
#include <string>
#include <memory>

#include "ir/Layout.h"
#include "ir/Index.h"
#include "ir/Operation.h"

namespace neurun
{
namespace ir
{

// To support ValueSwappable, Element doesn't have members which are classes
// as value(or can have members which are classes as value and the classes
// support Swappable)
struct Element
{
  OperationIndex index;
  const Operation *node;

  Element(const OperationIndex *i, const Operation *n) : index{*i}, node{n}
  {
    // DO NOTHING
  }
};

class OpSequence
{
public:
  explicit OpSequence(Layout layout);
  OpSequence(const OpSequence &) = delete;

public:
  void accept(OperationVisitor &v) const;

public:
  const OperandIndexSequence &getInputs() const { return _inputs; }
  const OperandIndexSequence &getOutputs() const { return _outputs; }
  void setInputs(const OperandIndexSequence &indexes) { _inputs = indexes; }
  void setOutputs(const OperandIndexSequence &indexes) { _outputs = indexes; }
  void replaceInput(const OperandIndex &from, const OperandIndex &to) { _inputs.replace(from, to); }
  void replaceOutput(const OperandIndex &from, const OperandIndex &to)
  {
    _outputs.replace(from, to);
  }

  void appendOperation(const OperationIndex &index, const Operation &node)
  {
    _operations.emplace_back(&index, &node);
  }

  std::vector<Element> &operations(void) { return _operations; }

  const std::vector<Element> &operations(void) const { return _operations; }

  uint32_t size(void) const { return _operations.size(); }

  // TODO: Impl Dumper instead of this method
  std::string getStr(void) const;

public:
  void remove(const OperationIndex &index);

public:
  Layout getLayout() const { return _layout; }

public:
  std::vector<Element>::const_iterator begin() const { return _operations.begin(); }
  std::vector<Element>::const_iterator end() const { return _operations.end(); }

private:
  bool exist(const OperationIndex &index) const;

private:
  OperandIndexSequence _inputs;
  OperandIndexSequence _outputs;
  std::vector<Element> _operations;

private:
  Layout _layout;
};

} // namespace ir
} // namespace neurun

#endif // __NEURUN_IR_OP_SEQUENCE_H__
