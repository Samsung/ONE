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

#ifndef __ONERT_IR_OP_SEQUENCE_H__
#define __ONERT_IR_OP_SEQUENCE_H__

#include <vector>
#include <string>
#include <memory>

#include "ir/Layout.h"
#include "ir/Index.h"
#include "ir/Operation.h"

namespace onert
{
namespace ir
{

class Operations;

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
  void replaceInputs(const OperandIndex &from, const OperandIndex &to)
  {
    _inputs.replace(from, to);
  }
  void replaceOutputs(const OperandIndex &from, const OperandIndex &to)
  {
    _outputs.replace(from, to);
  }

  void appendOperation(const OperationIndex &index) { _operations.emplace_back(index); }

  std::vector<OperationIndex> &operations(void) { return _operations; }

  const std::vector<OperationIndex> &operations(void) const { return _operations; }

  uint32_t size(void) const { return _operations.size(); }

public:
  void remove(const OperationIndex &index);

  bool exist(const OperationIndex &index) const;

public:
  Layout getLayout() const { return _layout; }

public:
  std::vector<OperationIndex>::const_iterator begin() const { return _operations.begin(); }
  std::vector<OperationIndex>::const_iterator end() const { return _operations.end(); }

public:
  /**
   * @brief Set @c true if any operation in this opSequence has dynamic input
   *        or dynamic output;
   *        @c false if all operations' inputs and outputs are static tensors
   */
  void hasDynamicTensor(bool hasDynamicTensor) { _has_dynamic_tensor = hasDynamicTensor; }
  bool hasDynamicTensor() const { return _has_dynamic_tensor; }

private:
  OperandIndexSequence _inputs;
  OperandIndexSequence _outputs;
  std::vector<OperationIndex> _operations;

private:
  Layout _layout;

private:
  bool _has_dynamic_tensor;
};

std::string getStrFromOpSeq(const OpSequence &op_seq, const Operations &operations);

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_OP_SEQUENCE_H__
