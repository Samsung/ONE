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

#include "ir/Operation.h"

#include <cassert>

namespace onert
{
namespace ir
{

Operation::Operation(OperandConstraint input_constr, const OperandIndexSequence &inputs,
                     const OperandIndexSequence &outputs, OperandConstraint output_constr)
  : _input_constr{input_constr}, _output_constr{output_constr}
{
  setInputs(inputs);
  setOutputs(outputs);
}

Operation::Operation(OperandConstraint input_constr, OperandConstraint output_constr)
  : _input_constr{input_constr}, _output_constr{output_constr}
{
}

Operation::~Operation() = default;

void Operation::setInputs(const OperandIndexSequence &indexes)
{
  if (!_input_constr.check(indexes.size()))
    throw std::runtime_error{"Invalid number of input tensors for this operation."};
  _inputs = indexes;
}

void Operation::setOutputs(const OperandIndexSequence &indexes)
{
  if (!_output_constr.check(indexes.size()))
    throw std::runtime_error{"Invalid number of output tensors for this operation."};
  _outputs = indexes;
}

void Operation::replaceInputs(const OperandIndex &from, const OperandIndex &to)
{
  _inputs.replace(from, to);
}

void Operation::replaceOutputs(const OperandIndex &from, const OperandIndex &to)
{
  _outputs.replace(from, to);
}

} // namespace ir
} // namespace onert
