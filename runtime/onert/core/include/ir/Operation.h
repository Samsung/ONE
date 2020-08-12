/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_OPERATION_H__
#define __ONERT_IR_OPERATION_H__

#include <memory>
#include <cassert>

#include "ir/OpCode.h"
#include "ir/Operand.h"
#include "ir/OperandIndexSequence.h"
#include "ir/OperandConstraint.h"

namespace onert
{
namespace ir
{

struct OperationVisitor;

class Operation
{
public:
  Operation(OperandConstraint input_constr, const OperandIndexSequence &inputs,
            const OperandIndexSequence &outputs);
  explicit Operation(OperandConstraint input_constr);

  Operation(const Operation &) = default;
  Operation(Operation &&) = default;
  Operation &operator=(const Operation &) = default;
  Operation &operator=(Operation &&) = default;

  virtual ~Operation();

public:
  virtual void accept(OperationVisitor &v) const = 0;
  virtual std::string name() const { return std::string{toString(opcode())}; }
  virtual OpCode opcode() const = 0;

public:
  void replaceInputs(const OperandIndex &from, const OperandIndex &to);
  void replaceOutputs(const OperandIndex &from, const OperandIndex &to);
  OperandIndexSequence &getInputs() { return _inputs; }
  const OperandIndexSequence &getInputs() const { return _inputs; }
  const OperandIndexSequence &getOutputs() const { return _outputs; }
  // It's for only input/output tensors but const data.
  void setInputs(const OperandIndexSequence &indexes);
  void setOutputs(const OperandIndexSequence &indexes);

  struct UniqueID
  {
    SubgraphIndex subg_ind;
    OpSequenceIndex op_seq_ind;
    OperationIndex op_ind;

    void set(SubgraphIndex subg, OpSequenceIndex op_seq, OperationIndex op)
    {
      subg_ind = subg;
      op_seq_ind = op_seq;
      op_ind = op;
    }

    bool isSame(SubgraphIndex subg, OpSequenceIndex op_seq, OperationIndex op)
    {
      assert(subg.valid() && op_seq.valid() && op.valid());
      assert(subg_ind.valid() && op_seq_ind.valid() && op_ind.valid());
      return subg == subg_ind && op_seq == op_seq_ind && op == op_ind;
    }
  };
  UniqueID id;

private:
  OperandConstraint _input_constr;
  OperandIndexSequence _inputs;
  OperandIndexSequence _outputs;
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_OPERATION_H__
