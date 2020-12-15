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

#ifndef __ANN_IR_OPERATION_H__
#define __ANN_IR_OPERATION_H__

#include "ANN/IR/OperandID.h"

#include <initializer_list>
#include <vector>

namespace ann
{

class Operation
{
public:
  enum class Code
  {
#define ANN_OPERATION(TAG, VALUE) TAG,
#include "Operation.def"
#undef ANN_OPERATION
  };

public:
  Operation(const Code &code, std::initializer_list<OperandID> inputs,
            std::initializer_list<OperandID> outputs)
    : _code{code}, _inputs{inputs}, _outputs{outputs}
  {
    // DO NOTHING
  }

public:
  const Code &code(void) const { return _code; }
  const std::vector<OperandID> &inputs(void) const { return _inputs; }
  const std::vector<OperandID> &outputs(void) const { return _outputs; }

private:
  Code _code;
  std::vector<OperandID> _inputs;
  std::vector<OperandID> _outputs;
};

} // namespace ann

#endif // __ANN_IR_OPERATION_H__
